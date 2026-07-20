"""Truthful inventory and explicit, receipt-backed heavy-byte cleanup."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
import errno
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
import uuid

from temper_ml.app_services._records import (
    require_no_conflicting_logical_revision,
    write_record_idempotently,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.runs import validated_run_lifecycle
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import CompatibilityGroup
from temper_ml.domain.datasets import DatasetVersion
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactAvailability,
    AvailabilityState,
    BundleManifest,
    StorageReference,
)
from temper_ml.domain.local_use import AdapterExport
from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.projects import Project
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    identity_fields,
    parse_identity,
    record_reference,
    require_identifier,
)
from temper_ml.domain.retention import (
    CleanupObjectReceipt,
    CleanupObjectStatus,
    CleanupOutcome,
    CleanupReceipt,
    require_cleanup_logical_key,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.runs import ResolvedRuntimeRequest, Run
from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    ensure_safe_directory,
    is_link_or_reparse,
    require_safe_directory,
    require_safe_regular_file,
    safe_path_stat,
    same_file_object,
)
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest
from temper_ml.store.safe_io import SafeIoError
from temper_ml.runtime.artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactIntegrityExpectation,
    EXPORT_BUNDLE_PREFIX,
    EXPORT_MANIFEST_MEMBER,
    verify_artifact_bundle,
)
from temper_ml.runtime.fixture_adapter import FIXTURE_ARTIFACT_MEMBERS
from temper_ml.runtime.ownership import (
    RunOwnershipError,
    claim_released_run_ownership,
    released_run_claim_identity,
)


RUNTIME_OUTPUT_DIRECTORY = ".temper-fixture-output"
CLEANUP_LOCK_LOGICAL_KEY = "runtime-ownership/cleanup/lease.lock"
INVENTORY_PROJECTION = HashProjection("retention.inventory", "v1")
ENTRY_PROJECTION = HashProjection("retention.inventory_entry", "v1")
PHYSICAL_GROUP_PROJECTION = HashProjection("retention.physical_group", "v1")
CLEANUP_PLAN_PROJECTION = HashProjection("retention.cleanup_plan", "v1")
_EXECUTION_ID = re.compile(r"^cleanup-execution-[0-9a-f]{32}$")
_HASH_CHUNK_BYTES = 1024 * 1024


class RetentionDefault(str, Enum):
    FULL = "full"


class ByteClass(str, Enum):
    CHECKPOINT = "checkpoint"
    FINAL_ADAPTER = "final_adapter"
    EXPORT_BUNDLE = "export_bundle"
    STAGING_CACHE = "staging_cache"
    RUNTIME_CONTROL = "runtime_control"
    DEBUG_EVIDENCE = "debugging_evidence"
    UNKNOWN = "unknown"

    @property
    def deletable(self) -> bool:
        return self in {
            ByteClass.CHECKPOINT,
            ByteClass.FINAL_ADAPTER,
            ByteClass.EXPORT_BUNDLE,
            ByteClass.STAGING_CACHE,
            ByteClass.DEBUG_EVIDENCE,
        }


class CleanupImpact(str, Enum):
    RESUMABILITY = "resumability"
    INSPECTABILITY = "inspectability"
    FINAL_ARTIFACT_AVAILABILITY = "final_artifact_availability"
    CACHE_CONVENIENCE = "cache_convenience"
    DEBUGGING_EVIDENCE = "debugging_evidence"
    SHARED_REFERENCE = "shared_reference"


@dataclass(frozen=True)
class InventoryEntry:
    """One portable logical link to verified local heavy bytes."""

    entry_id: str
    logical_key: str
    byte_class: ByteClass
    byte_count: int
    content_identity: ContentIdentity
    physical_group_id: str
    local_reference_count: int
    external_reference_count: int
    deletable: bool
    impacts: tuple[CleanupImpact, ...]
    subjects: tuple[RecordReference, ...]
    _path: Path = field(repr=False, compare=False)
    _signature: tuple[int, int, int, int, int] = field(repr=False, compare=False)

    def to_view(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "logical_key": self.logical_key,
            "byte_class": self.byte_class.value,
            "byte_count": self.byte_count,
            "content_identity": identity_fields(self.content_identity),
            "physical_group_id": self.physical_group_id,
            "local_reference_count": self.local_reference_count,
            "external_reference_count": self.external_reference_count,
            "deletable": self.deletable,
            "impacts": [impact.value for impact in self.impacts],
            "subjects": [subject.to_dict() for subject in self.subjects],
        }


@dataclass(frozen=True)
class StorageInventory:
    """Exact non-canonical byte inventory with a full-retention default."""

    inventory_identity: ContentIdentity
    entries: tuple[InventoryEntry, ...]

    @property
    def logical_bytes(self) -> int:
        return sum(entry.byte_count for entry in self.entries)

    @property
    def physical_bytes(self) -> int:
        groups: dict[str, int] = {}
        for entry in self.entries:
            groups.setdefault(entry.physical_group_id, entry.byte_count)
        return sum(groups.values())

    @property
    def reclaimable_physical_bytes(self) -> int:
        groups: dict[str, tuple[int, int]] = {}
        for entry in self.entries:
            groups.setdefault(
                entry.physical_group_id,
                (entry.byte_count, entry.external_reference_count),
            )
        deletable_groups = {
            entry.physical_group_id for entry in self.entries if entry.deletable
        }
        protected_groups = {
            entry.physical_group_id for entry in self.entries if not entry.deletable
        }
        return sum(
            size
            for group_id, (size, external) in groups.items()
            if external == 0
            and group_id in deletable_groups
            and group_id not in protected_groups
        )

    def to_view(self) -> dict[str, object]:
        counts: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            counts[entry.byte_class.value] += entry.byte_count
        return {
            "schema_version": "v1",
            "retention_default": RetentionDefault.FULL.value,
            "inventory_identity": identity_fields(self.inventory_identity),
            "entry_count": len(self.entries),
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
            "reclaimable_physical_bytes": self.reclaimable_physical_bytes,
            "byte_classes": dict(sorted(counts.items())),
            "entries": [entry.to_view() for entry in self.entries],
        }


@dataclass(frozen=True)
class CleanupPlan:
    """Exact selection and consequence projection; execution remains separate."""

    plan_id: str
    execution_id: str
    plan_identity: ContentIdentity
    inventory: StorageInventory
    selected_entry_ids: tuple[str, ...]
    selected_entries: tuple[InventoryEntry, ...]
    physical_group_ids_freed: tuple[str, ...]
    logical_bytes_selected: int
    physical_bytes_freed: int
    deleted_byte_classes: tuple[str, ...]
    retained_byte_classes: tuple[str, ...]
    impact_categories: tuple[CleanupImpact, ...]
    affected_subjects: tuple[RecordReference, ...]

    def to_view(self) -> dict[str, object]:
        warnings = []
        for impact in self.impact_categories:
            affected = tuple(
                entry.entry_id
                for entry in self.selected_entries
                if impact in entry.impacts
                or (
                    impact is CleanupImpact.SHARED_REFERENCE
                    and entry.physical_group_id not in self.physical_group_ids_freed
                )
            )
            if affected:
                warnings.append(
                    {
                        "category": impact.value,
                        "entry_ids": list(affected),
                    }
                )
        return {
            "schema_version": "v1",
            "plan_id": self.plan_id,
            "execution_id": self.execution_id,
            "plan_identity": identity_fields(self.plan_identity),
            "inventory_identity": identity_fields(self.inventory.inventory_identity),
            "retention_default": RetentionDefault.FULL.value,
            "requires_confirmation": True,
            "selected_entry_ids": list(self.selected_entry_ids),
            "logical_bytes_selected": self.logical_bytes_selected,
            "physical_bytes_freed": self.physical_bytes_freed,
            "deleted_byte_classes": list(self.deleted_byte_classes),
            "retained_byte_classes": list(self.retained_byte_classes),
            "warnings": warnings,
            "affected_subjects": [
                subject.to_dict() for subject in self.affected_subjects
            ],
            "entries": [entry.to_view() for entry in self.selected_entries],
        }


@dataclass(frozen=True)
class _ObservedFile:
    path: Path
    logical_key: str
    byte_class: ByteClass
    byte_count: int
    content_identity: ContentIdentity
    subjects: tuple[RecordReference, ...]
    impacts: tuple[CleanupImpact, ...]
    deletable: bool
    signature: tuple[int, int, int, int, int]
    link_count: int


@dataclass(frozen=True)
class _StableFileSnapshot:
    content_identity: ContentIdentity
    byte_count: int
    signature: tuple[int, int, int, int, int]


class _RemovalError(ApplicationServiceError):
    """Removal failure with an evidence-safe object disposition."""

    def __init__(self, code: str, status: CleanupObjectStatus) -> None:
        self.status = status
        super().__init__(code)


@dataclass(frozen=True)
class _CleanupIntentObject:
    entry_id: str
    logical_key: str
    byte_class: ByteClass
    byte_count: int
    content_identity: ContentIdentity
    physical_group_id: str
    subjects: tuple[RecordReference, ...]
    impacts: tuple[CleanupImpact, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "logical_key": self.logical_key,
            "byte_class": self.byte_class.value,
            "byte_count": self.byte_count,
            "content_identity": identity_fields(self.content_identity),
            "physical_group_id": self.physical_group_id,
            "subjects": [_event_reference(subject) for subject in self.subjects],
            "impacts": [impact.value for impact in self.impacts],
        }


@dataclass(frozen=True)
class _CleanupIntent:
    execution_id: str
    receipt_id: str
    project: RecordReference
    inventory_identity: ContentIdentity
    plan_identity: ContentIdentity
    selected_entry_ids: tuple[str, ...]
    objects: tuple[_CleanupIntentObject, ...]
    physical_group_ids_freed: tuple[str, ...]
    impact_categories: tuple[CleanupImpact, ...]
    affected_subjects: tuple[RecordReference, ...]

    @property
    def token(self) -> str:
        return self.execution_id.removeprefix("cleanup-execution-")

    @property
    def stream_id(self) -> str:
        return f"cleanup-{self.token}"

    def to_payload(self) -> dict[str, object]:
        return {
            "intent_schema_version": "v1",
            "execution_id": self.execution_id,
            "receipt_id": self.receipt_id,
            "project": _event_reference(self.project),
            "inventory_identity": identity_fields(self.inventory_identity),
            "plan_identity": identity_fields(self.plan_identity),
            "selected_entry_ids": list(self.selected_entry_ids),
            "objects": [item.to_payload() for item in self.objects],
            "physical_group_ids_freed": list(self.physical_group_ids_freed),
            "impact_categories": [item.value for item in self.impact_categories],
            "affected_subjects": [
                _event_reference(item) for item in self.affected_subjects
            ],
        }


class RetentionService:
    """Inventory and remove only explicit verified files below one fixed root."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        _remove_file: Callable[[Path], None] | None = None,
        _execution_id_factory: Callable[[], str] | None = None,
        _hash_chunk_size: int = _HASH_CHUNK_BYTES,
    ) -> None:
        self.project_root = Path(project_root)
        self.store = TypedEvidenceStore(self.project_root)
        self._cleanup_lock_held = False
        self._remove_file = _remove_file if _remove_file is not None else os.unlink
        self._execution_id_factory = (
            _execution_id_factory
            if _execution_id_factory is not None
            else lambda: f"cleanup-execution-{uuid.uuid4().hex}"
        )
        self._hash_chunk_size = _hash_chunk_size
        if not callable(self._remove_file):
            raise ApplicationServiceError("cleanup_remove_strategy_invalid")
        if not callable(self._execution_id_factory):
            raise ApplicationServiceError("cleanup_execution_id_factory_invalid")
        if (
            isinstance(self._hash_chunk_size, bool)
            or not isinstance(self._hash_chunk_size, int)
            or self._hash_chunk_size <= 0
        ):
            raise ApplicationServiceError("storage_hash_chunk_size_invalid")

    @contextmanager
    def _claim_cleanup_execution(self) -> Iterator[None]:
        """Serialize cleanup execution and interruption recovery for one project."""

        if self._cleanup_lock_held:
            yield
            return
        lock_path = self._runtime_root().joinpath(
            *PurePosixPath(CLEANUP_LOCK_LOGICAL_KEY).parts
        )
        try:
            ensure_safe_directory(lock_path.parent)
            existing = safe_path_stat(lock_path, allow_missing=True)
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ApplicationServiceError("cleanup_lock_invalid")
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(lock_path, flags, 0o600)
        except ApplicationServiceError:
            raise
        except (OSError, SafeIoError, UnsafeFilesystemPath):
            raise ApplicationServiceError("cleanup_lock_unavailable") from None
        with os.fdopen(descriptor, "r+b") as handle:
            locked = False
            try:
                opened = os.fstat(handle.fileno())
                current = require_safe_regular_file(lock_path)
                if (
                    is_link_or_reparse(opened)
                    or not stat.S_ISREG(opened.st_mode)
                    or not same_file_object(opened, current)
                    or opened.st_nlink != 1
                ):
                    raise ApplicationServiceError("cleanup_lock_invalid")
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                elif size != 1:
                    raise ApplicationServiceError("cleanup_lock_invalid")
                handle.seek(0)
                _lock_cleanup_handle(handle)
                locked = True
                opened = os.fstat(handle.fileno())
                current = require_safe_regular_file(lock_path)
                handle.seek(0)
                if (
                    not same_file_object(opened, current)
                    or opened.st_nlink != 1
                    or opened.st_size != 1
                    or handle.read(1) != b"\0"
                ):
                    raise ApplicationServiceError("cleanup_lock_invalid")
            except ApplicationServiceError:
                if locked:
                    try:
                        handle.seek(0)
                        _unlock_cleanup_handle(handle)
                    except OSError:
                        pass
                raise
            except (OSError, SafeIoError, UnsafeFilesystemPath):
                if locked:
                    try:
                        handle.seek(0)
                        _unlock_cleanup_handle(handle)
                    except OSError:
                        pass
                raise ApplicationServiceError("cleanup_lock_unavailable") from None
            self._cleanup_lock_held = True
            try:
                yield
            finally:
                self._cleanup_lock_held = False
                try:
                    handle.seek(0)
                    _unlock_cleanup_handle(handle)
                except OSError:
                    pass

    def inventory(self, *, reconcile_pending: bool = True) -> StorageInventory:
        """Build one link-safe, content-verified snapshot of all runtime bytes."""

        if reconcile_pending:
            self._reconcile_pending()
        try:
            self.store.verify()
            records = tuple(stored.record for stored in self.store.iter_records())
            streams = self.store.iter_streams()
            manifests = self.store.iter_bundle_manifests()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        root = self._runtime_root()
        released_run_ids = self._released_run_ids(records)
        try:
            if safe_path_stat(root, allow_missing=True) is None:
                identity = content_identity(INVENTORY_PROJECTION, {"entries": []})
                return StorageInventory(identity, ())
            require_safe_directory(root)
            paths = tuple(
                path
                for path in _enumerate_regular_files(root)
                if path.relative_to(root).as_posix() != CLEANUP_LOCK_LOGICAL_KEY
            )
            observations = tuple(
                self._observe(
                    path,
                    root,
                    records,
                    streams,
                    manifests,
                    released_run_ids,
                )
                for path in paths
            )
        except ApplicationServiceError:
            raise
        except (OSError, SafeIoError, UnsafeFilesystemPath):
            raise ApplicationServiceError("storage_inventory_unsafe") from None
        groups: dict[tuple[int, int], list[_ObservedFile]] = defaultdict(list)
        for observation in observations:
            groups[observation.signature[:2]].append(observation)
        entries: list[InventoryEntry] = []
        for group in sorted(groups.values(), key=lambda value: value[0].logical_key):
            group = sorted(group, key=lambda item: item.logical_key)
            if (
                len({item.byte_count for item in group}) != 1
                or len({item.content_identity for item in group}) != 1
            ):
                raise ApplicationServiceError("storage_inventory_unstable")
            local_count = len(group)
            external_count = max(
                0, max(item.link_count for item in group) - local_count
            )
            group_identity = content_identity(
                PHYSICAL_GROUP_PROJECTION,
                {
                    "logical_keys": [item.logical_key for item in group],
                    "byte_count": group[0].byte_count,
                    "content_identity": identity_fields(group[0].content_identity),
                    "external_reference_count": external_count,
                },
            )
            physical_group_id = f"physical-{group_identity.value[:24]}"
            for observation in group:
                entry_identity = content_identity(
                    ENTRY_PROJECTION,
                    {
                        "logical_key": observation.logical_key,
                        "byte_class": observation.byte_class.value,
                        "byte_count": observation.byte_count,
                        "content_identity": identity_fields(
                            observation.content_identity
                        ),
                        "physical_group_id": physical_group_id,
                        "local_reference_count": local_count,
                        "external_reference_count": external_count,
                        "subjects": [
                            subject.to_dict() for subject in observation.subjects
                        ],
                    },
                )
                entries.append(
                    InventoryEntry(
                        entry_id=f"entry-{entry_identity.value[:24]}",
                        logical_key=observation.logical_key,
                        byte_class=observation.byte_class,
                        byte_count=observation.byte_count,
                        content_identity=observation.content_identity,
                        physical_group_id=physical_group_id,
                        local_reference_count=local_count,
                        external_reference_count=external_count,
                        deletable=observation.deletable,
                        impacts=observation.impacts,
                        subjects=observation.subjects,
                        _path=observation.path,
                        _signature=observation.signature,
                    )
                )
        ordered = tuple(sorted(entries, key=lambda item: item.entry_id))
        identity = content_identity(
            INVENTORY_PROJECTION,
            {"entries": [entry.to_view() for entry in ordered]},
        )
        return StorageInventory(identity, ordered)

    def plan(
        self,
        entry_ids: Iterable[str],
        *,
        execution_id: str | None = None,
    ) -> CleanupPlan:
        """Calculate exact effects without selecting or deleting anything implicitly."""

        if isinstance(entry_ids, (str, bytes)):
            raise ApplicationServiceError("cleanup_selection_invalid")
        supplied_ids = tuple(entry_ids)
        if not supplied_ids or any(
            not isinstance(entry_id, str) or not entry_id for entry_id in supplied_ids
        ):
            raise ApplicationServiceError("cleanup_selection_required")
        selected_ids = tuple(sorted(supplied_ids))
        if len(set(selected_ids)) != len(selected_ids):
            raise ApplicationServiceError("cleanup_selection_duplicate")
        inventory = self.inventory()
        by_id = {entry.entry_id: entry for entry in inventory.entries}
        if any(entry_id not in by_id for entry_id in selected_ids):
            raise ApplicationServiceError("cleanup_selection_unknown")
        selected = tuple(by_id[entry_id] for entry_id in selected_ids)
        if any(not entry.deletable for entry in selected):
            raise ApplicationServiceError("cleanup_selection_protected")
        selected_set = set(selected_ids)
        groups: dict[str, list[InventoryEntry]] = defaultdict(list)
        for entry in inventory.entries:
            groups[entry.physical_group_id].append(entry)
        freed_groups = tuple(
            sorted(
                group_id
                for group_id, entries in groups.items()
                if {entry.entry_id for entry in entries} <= selected_set
                and entries[0].external_reference_count == 0
            )
        )
        physical_bytes = sum(
            groups[group_id][0].byte_count for group_id in freed_groups
        )
        impacts = {impact for entry in selected for impact in entry.impacts}
        if any(entry.physical_group_id not in freed_groups for entry in selected):
            impacts.add(CleanupImpact.SHARED_REFERENCE)
        affected = _sorted_references(
            subject for entry in selected for subject in entry.subjects
        )
        plan_identity = content_identity(
            CLEANUP_PLAN_PROJECTION,
            {
                "inventory_identity": identity_fields(inventory.inventory_identity),
                "selected_entry_ids": list(selected_ids),
            },
        )
        selected_classes = {entry.byte_class.value for entry in selected}
        retained_classes = {
            entry.byte_class.value
            for entry in inventory.entries
            if entry.entry_id not in selected_set
        }
        selected_execution_id = (
            self._execution_id_factory() if execution_id is None else execution_id
        )
        if (
            not isinstance(selected_execution_id, str)
            or _EXECUTION_ID.fullmatch(selected_execution_id) is None
        ):
            raise ApplicationServiceError("cleanup_execution_id_invalid")
        return CleanupPlan(
            plan_id=f"cleanup-plan-{plan_identity.value[:24]}",
            execution_id=selected_execution_id,
            plan_identity=plan_identity,
            inventory=inventory,
            selected_entry_ids=selected_ids,
            selected_entries=selected,
            physical_group_ids_freed=freed_groups,
            logical_bytes_selected=sum(entry.byte_count for entry in selected),
            physical_bytes_freed=physical_bytes,
            deleted_byte_classes=tuple(sorted(selected_classes)),
            retained_byte_classes=tuple(sorted(retained_classes)),
            impact_categories=tuple(sorted(impacts, key=lambda item: item.value)),
            affected_subjects=affected,
        )

    def execute(self, plan: CleanupPlan, *, confirm: bool) -> CleanupReceipt:
        """Execute one current plan under a unique recoverable execution identity."""

        if not isinstance(plan, CleanupPlan):
            raise ApplicationServiceError("cleanup_plan_invalid")
        if confirm is not True:
            raise ApplicationServiceError("cleanup_confirmation_required")
        if _EXECUTION_ID.fullmatch(plan.execution_id) is None:
            raise ApplicationServiceError("cleanup_execution_id_invalid")
        with self._claim_cleanup_execution():
            return self._execute_locked(plan)

    def _execute_locked(self, plan: CleanupPlan) -> CleanupReceipt:
        self._reconcile_pending()
        receipt_id = _receipt_id(plan.execution_id)
        existing = self._receipt_by_id(receipt_id)
        if existing is not None:
            if (
                existing.execution_id != plan.execution_id
                or existing.plan_identity != plan.plan_identity
                or existing.inventory_identity != plan.inventory.inventory_identity
            ):
                raise ApplicationServiceError("cleanup_execution_conflict")
            return existing
        current = self.inventory(reconcile_pending=False)
        if current.inventory_identity != plan.inventory.inventory_identity:
            raise ApplicationServiceError("cleanup_plan_stale")
        current_by_id = {entry.entry_id: entry for entry in current.entries}
        if any(entry_id not in current_by_id for entry_id in plan.selected_entry_ids):
            raise ApplicationServiceError("cleanup_plan_stale")
        selected = tuple(current_by_id[item] for item in plan.selected_entry_ids)
        planned_by_id = {entry.entry_id: entry for entry in plan.selected_entries}
        if any(
            entry.to_view() != planned_by_id[entry.entry_id].to_view()
            or entry._signature != planned_by_id[entry.entry_id]._signature
            for entry in selected
        ):
            raise ApplicationServiceError("cleanup_plan_stale")
        with self._claim_producer_ownership(selected):
            return self._execute_owned(plan, current, selected)

    def _execute_owned(
        self,
        plan: CleanupPlan,
        current: StorageInventory,
        selected: tuple[InventoryEntry, ...],
    ) -> CleanupReceipt:
        intent = self._intent_from_plan(plan, selected)
        self._append_cleanup(
            intent.stream_id,
            f"{intent.execution_id}-started",
            "cleanup_started",
            intent.to_payload(),
        )
        statuses: dict[str, CleanupObjectStatus] = {
            entry.entry_id: CleanupObjectStatus.NOT_ATTEMPTED for entry in selected
        }
        groups: dict[str, tuple[InventoryEntry, ...]] = {
            group_id: tuple(
                entry
                for entry in current.entries
                if entry.physical_group_id == group_id
            )
            for group_id in {entry.physical_group_id for entry in selected}
        }
        verified_groups: set[str] = set()
        removed_by_group: dict[str, int] = defaultdict(int)
        failure_code: str | None = None
        removal_ambiguous = False
        try:
            self._prepare_safety_fences(intent)
            self._append_cleanup(
                intent.stream_id,
                f"{intent.execution_id}-prepared",
                "cleanup_prepared",
                {"execution_id": intent.execution_id},
            )
        except (ApplicationServiceError, EvidenceError) as exc:
            failure_code = getattr(exc, "code", "cleanup_evidence_persistence_failed")
            self._record_failure(intent, failure_code)
        if failure_code is None:
            for entry in selected:
                if entry.physical_group_id not in verified_groups:
                    try:
                        self._verify_group_unchanged(groups[entry.physical_group_id])
                    except ApplicationServiceError as exc:
                        for selected_entry in selected:
                            if (
                                selected_entry.physical_group_id
                                == entry.physical_group_id
                            ):
                                statuses[selected_entry.entry_id] = (
                                    CleanupObjectStatus.AMBIGUOUS
                                )
                        failure_code = exc.code
                        self._record_failure(intent, failure_code, entry.entry_id)
                        break
                    verified_groups.add(entry.physical_group_id)
                try:
                    self._append_cleanup(
                        intent.stream_id,
                        f"{intent.execution_id}-intent-{entry.entry_id}",
                        "cleanup_object_deletion_intent",
                        {
                            "entry_id": entry.entry_id,
                            "content_identity": identity_fields(entry.content_identity),
                        },
                    )
                except ApplicationServiceError as exc:
                    statuses[entry.entry_id] = CleanupObjectStatus.FAILED
                    failure_code = exc.code
                    self._record_failure(intent, failure_code, entry.entry_id)
                    break
                try:
                    self._unlink_verified(
                        entry,
                        expected_link_count=(
                            entry._signature[-1]
                            - removed_by_group[entry.physical_group_id]
                        ),
                    )
                    statuses[entry.entry_id] = CleanupObjectStatus.REMOVED
                    removed_by_group[entry.physical_group_id] += 1
                except _RemovalError as exc:
                    statuses[entry.entry_id] = exc.status
                    failure_code = exc.code
                    removal_ambiguous = exc.status is CleanupObjectStatus.AMBIGUOUS
                    self._record_failure(intent, failure_code, entry.entry_id)
                    break
                try:
                    self._append_object_removed(intent, entry.entry_id)
                except ApplicationServiceError as exc:
                    failure_code = exc.code
                    self._record_failure(intent, failure_code, entry.entry_id)
                    break
        if removal_ambiguous:
            raise ApplicationServiceError("cleanup_reconciliation_required")
        return self._finalize_execution(intent, statuses, failure_code)

    def receipts(self, *, reconcile_pending: bool = True) -> tuple[CleanupReceipt, ...]:
        if reconcile_pending:
            self._reconcile_pending()
        try:
            return tuple(
                stored.record
                for stored in self.store.iter_records()
                if isinstance(stored.record, CleanupReceipt)
            )
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None

    def reconcile_pending(self) -> tuple[CleanupReceipt, ...]:
        """Recover interrupted cleanup only at an explicit mutation boundary."""

        try:
            self._reconcile_pending()
        except ApplicationServiceError as exc:
            if exc.code in {
                "project_not_found",
                "store_missing",
                "store_not_found",
                "store_root_missing",
            }:
                return ()
            raise
        return self.receipts(reconcile_pending=False)

    def _observe(
        self,
        path: Path,
        root: Path,
        records: tuple[TypedRecord, ...],
        streams: tuple[Any, ...],
        manifests: tuple[BundleManifest, ...],
        released_run_ids: frozenset[str],
    ) -> _ObservedFile:
        try:
            snapshot = _stream_file_snapshot(path, chunk_size=self._hash_chunk_size)
        except (OSError, SafeIoError, UnsafeFilesystemPath):
            raise ApplicationServiceError("storage_inventory_unstable") from None
        logical_key = path.relative_to(root).as_posix()
        classification = _classify(
            logical_key,
            records,
            streams,
            manifests,
            snapshot.content_identity,
            released_run_ids,
        )
        try:
            require_cleanup_logical_key(logical_key)
        except RecordValidationError:
            classification = _protected_unknown(classification[1])
        return _ObservedFile(
            path=path,
            logical_key=logical_key,
            byte_class=classification[0],
            byte_count=snapshot.byte_count,
            content_identity=snapshot.content_identity,
            subjects=classification[1],
            impacts=classification[2],
            deletable=classification[3],
            signature=snapshot.signature,
            link_count=max(1, snapshot.signature[-1]),
        )

    def _verify_group_unchanged(self, entries: tuple[InventoryEntry, ...]) -> None:
        """Revalidate every local link before removing any member of its group."""

        for entry in entries:
            try:
                snapshot = _stream_file_snapshot(
                    entry._path, chunk_size=self._hash_chunk_size
                )
            except (FileNotFoundError, SafeIoError, UnsafeFilesystemPath):
                raise ApplicationServiceError("cleanup_object_changed") from None
            if (
                snapshot.signature != entry._signature
                or snapshot.byte_count != entry.byte_count
                or snapshot.content_identity != entry.content_identity
            ):
                raise ApplicationServiceError("cleanup_object_changed")

    def _unlink_verified(
        self,
        entry: InventoryEntry,
        *,
        expected_link_count: int,
    ) -> None:
        expected_signature = (*entry._signature[:-1], expected_link_count)
        try:
            snapshot = _stream_file_snapshot(
                entry._path, chunk_size=self._hash_chunk_size
            )
            if (
                snapshot.signature != expected_signature
                or snapshot.byte_count != entry.byte_count
                or snapshot.content_identity != entry.content_identity
            ):
                raise _RemovalError(
                    "cleanup_object_changed", CleanupObjectStatus.AMBIGUOUS
                )
        except FileNotFoundError:
            raise _RemovalError(
                "cleanup_object_removal_ambiguous", CleanupObjectStatus.AMBIGUOUS
            ) from None
        except (OSError, SafeIoError, UnsafeFilesystemPath):
            raise _RemovalError(
                "cleanup_object_removal_ambiguous", CleanupObjectStatus.AMBIGUOUS
            ) from None
        try:
            self._remove_file(entry._path)
        except OSError:
            if self._entry_still_matches(
                entry, expected_link_count=expected_link_count
            ):
                raise _RemovalError(
                    "cleanup_object_remove_failed", CleanupObjectStatus.RETAINED
                ) from None
            raise _RemovalError(
                "cleanup_object_removal_ambiguous", CleanupObjectStatus.AMBIGUOUS
            ) from None
        try:
            remaining = safe_path_stat(entry._path, allow_missing=True)
        except (OSError, UnsafeFilesystemPath):
            raise _RemovalError(
                "cleanup_object_removal_ambiguous", CleanupObjectStatus.AMBIGUOUS
            ) from None
        if remaining is not None:
            if self._entry_still_matches(
                entry, expected_link_count=expected_link_count
            ):
                raise _RemovalError(
                    "cleanup_object_remove_failed", CleanupObjectStatus.RETAINED
                )
            raise _RemovalError(
                "cleanup_object_removal_ambiguous", CleanupObjectStatus.AMBIGUOUS
            )

    def _entry_still_matches(
        self,
        entry: InventoryEntry,
        *,
        expected_link_count: int,
    ) -> bool:
        try:
            snapshot = _stream_file_snapshot(
                entry._path, chunk_size=self._hash_chunk_size
            )
        except (FileNotFoundError, OSError, SafeIoError, UnsafeFilesystemPath):
            return False
        return (
            snapshot.signature == (*entry._signature[:-1], expected_link_count)
            and snapshot.byte_count == entry.byte_count
            and snapshot.content_identity == entry.content_identity
        )

    def _intent_from_plan(
        self,
        plan: CleanupPlan,
        selected: tuple[InventoryEntry, ...],
    ) -> _CleanupIntent:
        intent = _CleanupIntent(
            execution_id=plan.execution_id,
            receipt_id=_receipt_id(plan.execution_id),
            project=record_reference(self._project()),
            inventory_identity=plan.inventory.inventory_identity,
            plan_identity=plan.plan_identity,
            selected_entry_ids=plan.selected_entry_ids,
            objects=tuple(
                _CleanupIntentObject(
                    entry_id=entry.entry_id,
                    logical_key=entry.logical_key,
                    byte_class=entry.byte_class,
                    byte_count=entry.byte_count,
                    content_identity=entry.content_identity,
                    physical_group_id=entry.physical_group_id,
                    subjects=entry.subjects,
                    impacts=entry.impacts,
                )
                for entry in selected
            ),
            physical_group_ids_freed=plan.physical_group_ids_freed,
            impact_categories=plan.impact_categories,
            affected_subjects=plan.affected_subjects,
        )
        try:
            validated = _intent_from_payload(intent.to_payload())
        except (RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError("cleanup_plan_unrepresentable") from None
        if validated != intent:
            raise ApplicationServiceError("cleanup_plan_unrepresentable")
        return validated

    def _prepare_safety_fences(self, intent: _CleanupIntent) -> None:
        for _, artifact in self._intent_artifacts(intent):
            self._ensure_pending_availability(intent, artifact)
        for item in intent.objects:
            if item.byte_class is not ByteClass.CHECKPOINT:
                continue
            for subject in item.subjects:
                if subject.record_type != "run":
                    continue
                self.store.append_event(
                    f"run-{subject.logical_id}",
                    EventRequest(
                        f"{intent.execution_id}-{item.entry_id}-pending",
                        "run_checkpoint_cleanup_pending",
                        {
                            "execution_id": intent.execution_id,
                            "entry_id": item.entry_id,
                            "content_identity": identity_fields(item.content_identity),
                            "resume_available": False,
                        },
                    ),
                )

    def _ensure_pending_availability(
        self,
        intent: _CleanupIntent,
        artifact: Artifact,
    ) -> ArtifactAvailability:
        availability_id = _availability_id("pending", intent, artifact)
        existing = self._availability_by_id(availability_id)
        if existing is not None:
            if (
                existing.artifact != record_reference(artifact)
                or existing.state is not AvailabilityState.UNAVAILABLE
            ):
                raise ApplicationServiceError("cleanup_availability_conflict")
            return existing
        current = self._current_availability(artifact)
        pending = ArtifactAvailability(
            availability_id=availability_id,
            artifact=record_reference(artifact),
            state=AvailabilityState.UNAVAILABLE,
            available_byte_classes=(),
            storage_references=(),
            checkpoint_resumable=False,
            observed_content_identity=artifact.content_identity,
            supersedes=record_reference(current),
        )
        require_no_conflicting_logical_revision(
            self.store,
            pending,
            conflict_code="cleanup_availability_conflict",
        )
        write_record_idempotently(
            self.store,
            pending,
            conflict_code="cleanup_availability_conflict",
        )
        return pending

    def _ensure_terminal_availability(
        self,
        intent: _CleanupIntent,
        artifact: Artifact,
        *,
        removed: bool,
    ) -> ArtifactAvailability | None:
        pending = self._availability_by_id(
            _availability_id("pending", intent, artifact)
        )
        if pending is None:
            if removed:
                raise ApplicationServiceError("cleanup_artifact_fence_missing")
            return None
        if not removed:
            self._verify_complete_artifact_bundle(artifact)
        disposition = "removed" if removed else "restored"
        availability_id = _availability_id(disposition, intent, artifact)
        existing = self._availability_by_id(availability_id)
        if existing is not None:
            return existing
        if removed:
            state = AvailabilityState.REMOVED
            available_byte_classes: tuple[str, ...] = ()
            storage_references: tuple[StorageReference, ...] = ()
            checkpoint_resumable = False
            observed_content_identity = artifact.content_identity
        else:
            if pending.supersedes is None:
                raise ApplicationServiceError("cleanup_availability_conflict")
            try:
                prior = self.store.read_record(pending.supersedes).record
            except EvidenceError:
                raise ApplicationServiceError("cleanup_availability_conflict") from None
            if not isinstance(prior, ArtifactAvailability):
                raise ApplicationServiceError("cleanup_availability_conflict")
            state = prior.state
            available_byte_classes = prior.available_byte_classes
            storage_references = prior.storage_references
            checkpoint_resumable = prior.checkpoint_resumable
            observed_content_identity = prior.observed_content_identity
        update = ArtifactAvailability(
            availability_id=availability_id,
            artifact=record_reference(artifact),
            state=state,
            available_byte_classes=available_byte_classes,
            storage_references=storage_references,
            checkpoint_resumable=checkpoint_resumable,
            observed_content_identity=observed_content_identity,
            supersedes=record_reference(pending),
        )
        require_no_conflicting_logical_revision(
            self.store,
            update,
            conflict_code="cleanup_availability_conflict",
        )
        write_record_idempotently(
            self.store,
            update,
            conflict_code="cleanup_availability_conflict",
        )
        return update

    def _verify_complete_artifact_bundle(self, artifact: Artifact) -> None:
        run = self._read_exact_record(
            artifact.producing_run, Run, "cleanup_artifact_integrity_mismatch"
        )
        runtime_request = self._record_by_identity(
            ResolvedRuntimeRequest,
            run.request_identity,
            "cleanup_artifact_integrity_mismatch",
        )
        experiment = self._read_exact_record(
            run.experiment, Experiment, "cleanup_artifact_integrity_mismatch"
        )
        resolution = self._read_exact_record(
            runtime_request.recipe_resolution,
            RecipeResolution,
            "cleanup_artifact_integrity_mismatch",
        )
        dataset = self._record_by_identity(
            DatasetVersion,
            runtime_request.dataset_version_identity,
            "cleanup_artifact_integrity_mismatch",
        )
        model = self._read_exact_record(
            artifact.base_model_revision,
            BaseModelRevision,
            "cleanup_artifact_integrity_mismatch",
        )
        group = self._read_exact_record(
            experiment.compatibility_group,
            CompatibilityGroup,
            "cleanup_artifact_integrity_mismatch",
        )
        if (
            experiment.compatibility_group not in artifact.compatibility_groups
            or artifact.tokenizer_identity != model.tokenizer_identity
        ):
            raise ApplicationServiceError("cleanup_artifact_integrity_mismatch")
        try:
            expectation = ArtifactIntegrityExpectation(
                bundle_identity=artifact.content_identity,
                producing_run=run,
                runtime_request=runtime_request,
                experiment=experiment,
                recipe_resolution=resolution,
                dataset_version=dataset,
                base_model_revision=model,
                compatibility_group=group,
            )
            integrity = verify_artifact_bundle(
                self._runtime_root() / "artifacts" / artifact.artifact_id,
                expectation,
            )
            stored_manifest = self.store.read_bundle_manifest(artifact.content_identity)
        except ArtifactIntegrityError as exc:
            raise ApplicationServiceError(exc.code) from None
        except EvidenceError:
            raise ApplicationServiceError(
                "cleanup_artifact_integrity_mismatch"
            ) from None
        if (
            integrity.bundle_manifest != stored_manifest
            or integrity.evidence_identity != artifact.integrity_evidence
            or integrity.provenance_identity != artifact.provenance
        ):
            raise ApplicationServiceError("cleanup_artifact_integrity_mismatch")

    def _finalize_execution(
        self,
        intent: _CleanupIntent,
        statuses: Mapping[str, CleanupObjectStatus],
        failure_code: str | None,
    ) -> CleanupReceipt:
        first_failure = failure_code
        for item in intent.objects:
            if statuses[item.entry_id] is not CleanupObjectStatus.REMOVED:
                continue
            try:
                self._append_object_removed(intent, item.entry_id)
            except ApplicationServiceError as exc:
                first_failure = first_failure or exc.code
        availability_updates, safety_failure = self._settle_safety_fences(
            intent, statuses
        )
        first_failure = first_failure or safety_failure
        receipt = self._build_receipt(
            intent,
            statuses,
            first_failure,
            availability_updates,
        )
        try:
            require_no_conflicting_logical_revision(
                self.store,
                receipt,
                conflict_code="cleanup_receipt_conflict",
            )
            write_record_idempotently(
                self.store,
                receipt,
                conflict_code="cleanup_receipt_conflict",
            )
        except (ApplicationServiceError, EvidenceError) as exc:
            self._record_failure(
                intent,
                getattr(exc, "code", "cleanup_receipt_persistence_failed"),
            )
            raise ApplicationServiceError("cleanup_reconciliation_required") from None
        try:
            self._append_terminal(intent, receipt)
            self.store.verify()
        except (ApplicationServiceError, EvidenceError):
            raise ApplicationServiceError("cleanup_reconciliation_required") from None
        return receipt

    def _settle_safety_fences(
        self,
        intent: _CleanupIntent,
        statuses: Mapping[str, CleanupObjectStatus],
    ) -> tuple[tuple[ArtifactAvailability, ...], str | None]:
        first_failure: str | None = None
        availability_updates: list[ArtifactAvailability] = []
        for reference, artifact in self._intent_artifacts(intent):
            affected = tuple(
                item
                for item in intent.objects
                if item.byte_class is ByteClass.FINAL_ADAPTER
                and reference in item.subjects
            )
            removed = any(
                statuses[item.entry_id] is CleanupObjectStatus.REMOVED
                for item in affected
            )
            ambiguous = any(
                statuses[item.entry_id] is CleanupObjectStatus.AMBIGUOUS
                for item in affected
            )
            changed = not removed and any(
                statuses[item.entry_id] is not CleanupObjectStatus.REMOVED
                and not self._intent_object_still_matches(item)
                for item in affected
            )
            uncertain = not removed and (ambiguous or changed)
            if uncertain:
                first_failure = first_failure or (
                    "cleanup_object_changed" if changed else "cleanup_removal_ambiguous"
                )
                update = self._availability_by_id(
                    _availability_id("pending", intent, artifact)
                )
            else:
                try:
                    update = self._ensure_terminal_availability(
                        intent, artifact, removed=removed
                    )
                except (ApplicationServiceError, EvidenceError) as exc:
                    first_failure = first_failure or getattr(
                        exc, "code", "cleanup_evidence_persistence_failed"
                    )
                    update = self._availability_by_id(
                        _availability_id("pending", intent, artifact)
                    )
            if update is not None:
                availability_updates.append(update)
        for item in intent.objects:
            if item.byte_class is not ByteClass.CHECKPOINT:
                continue
            removed = statuses[item.entry_id] is CleanupObjectStatus.REMOVED
            if statuses[item.entry_id] is CleanupObjectStatus.AMBIGUOUS:
                first_failure = first_failure or "cleanup_removal_ambiguous"
                continue
            if not removed and not self._intent_object_still_matches(item):
                first_failure = first_failure or "cleanup_object_changed"
                continue
            for subject in item.subjects:
                if subject.record_type != "run":
                    continue
                try:
                    if not self._checkpoint_fence_exists(intent, item, subject):
                        continue
                except ApplicationServiceError as exc:
                    first_failure = first_failure or exc.code
                    continue
                event_type = (
                    "run_checkpoint_removed"
                    if removed
                    else "run_checkpoint_cleanup_cancelled"
                )
                try:
                    self.store.append_event(
                        f"run-{subject.logical_id}",
                        EventRequest(
                            (
                                f"{intent.execution_id}-{item.entry_id}-"
                                f"{'removed' if removed else 'cancelled'}"
                            ),
                            event_type,
                            {
                                "execution_id": intent.execution_id,
                                "entry_id": item.entry_id,
                                "content_identity": identity_fields(
                                    item.content_identity
                                ),
                                "resume_available": (
                                    not removed
                                    and CleanupImpact.RESUMABILITY in item.impacts
                                ),
                            },
                        ),
                    )
                except EvidenceError as exc:
                    first_failure = first_failure or exc.code
        return tuple(availability_updates), first_failure

    def _intent_object_still_matches(self, item: _CleanupIntentObject) -> bool:
        path = self._runtime_root().joinpath(*PurePosixPath(item.logical_key).parts)
        try:
            snapshot = _stream_file_snapshot(path, chunk_size=self._hash_chunk_size)
        except (FileNotFoundError, OSError, SafeIoError, UnsafeFilesystemPath):
            return False
        return (
            snapshot.byte_count == item.byte_count
            and snapshot.content_identity == item.content_identity
        )

    def _checkpoint_fence_exists(
        self,
        intent: _CleanupIntent,
        item: _CleanupIntentObject,
        subject: RecordReference,
    ) -> bool:
        stream_id = f"run-{subject.logical_id}"
        try:
            events = next(
                (
                    snapshot.events
                    for snapshot in self.store.iter_streams()
                    if snapshot.stream_id == stream_id
                ),
                (),
            )
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        key = f"{intent.execution_id}-{item.entry_id}-pending"
        matches = tuple(event for event in events if event.idempotency_key == key)
        expected_payload = {
            "execution_id": intent.execution_id,
            "entry_id": item.entry_id,
            "content_identity": identity_fields(item.content_identity),
            "resume_available": False,
        }
        if not matches:
            return False
        if (
            len(matches) != 1
            or matches[0].event_type != "run_checkpoint_cleanup_pending"
            or dict(matches[0].payload) != expected_payload
        ):
            raise ApplicationServiceError("cleanup_checkpoint_fence_conflict")
        return True

    def _build_receipt(
        self,
        intent: _CleanupIntent,
        statuses: Mapping[str, CleanupObjectStatus],
        failure_code: str | None,
        availability_updates: tuple[ArtifactAvailability, ...],
    ) -> CleanupReceipt:
        removed = tuple(
            item
            for item in intent.objects
            if statuses[item.entry_id] is CleanupObjectStatus.REMOVED
        )
        freed_groups = {
            group_id
            for group_id in intent.physical_group_ids_freed
            if all(
                statuses[item.entry_id] is CleanupObjectStatus.REMOVED
                for item in intent.objects
                if item.physical_group_id == group_id
            )
        }
        physical_representatives = {
            min(
                item.entry_id
                for item in intent.objects
                if item.physical_group_id == group_id
            )
            for group_id in freed_groups
        }
        if failure_code is None and len(removed) != len(intent.objects):
            failure_code = "cleanup_incomplete"
        if failure_code is None:
            outcome = CleanupOutcome.COMPLETED
        elif not removed:
            outcome = CleanupOutcome.FAILED
        else:
            outcome = CleanupOutcome.PARTIAL
        return CleanupReceipt(
            receipt_id=intent.receipt_id,
            execution_id=intent.execution_id,
            project=intent.project,
            inventory_identity=intent.inventory_identity,
            plan_identity=intent.plan_identity,
            outcome=outcome,
            selected_entry_ids=intent.selected_entry_ids,
            objects=tuple(
                CleanupObjectReceipt(
                    entry_id=item.entry_id,
                    logical_key=item.logical_key,
                    byte_class=item.byte_class.value,
                    byte_count=item.byte_count,
                    content_identity=item.content_identity,
                    status=statuses[item.entry_id],
                    physical_bytes_freed=(item.entry_id in physical_representatives),
                    subjects=item.subjects,
                )
                for item in intent.objects
            ),
            logical_bytes_removed=sum(item.byte_count for item in removed),
            physical_bytes_freed=sum(
                item.byte_count
                for item in removed
                if item.entry_id in physical_representatives
            ),
            impact_categories=tuple(
                impact.value for impact in intent.impact_categories
            ),
            affected_subjects=intent.affected_subjects,
            availability_updates=_sorted_references(
                record_reference(update) for update in availability_updates
            ),
            failure_code=failure_code,
        )

    def _intent_artifacts(
        self, intent: _CleanupIntent
    ) -> tuple[tuple[RecordReference, Artifact], ...]:
        references = _sorted_references(
            subject
            for item in intent.objects
            if item.byte_class is ByteClass.FINAL_ADAPTER
            for subject in item.subjects
            if subject.record_type == "artifact"
        )
        values: list[tuple[RecordReference, Artifact]] = []
        for reference in references:
            try:
                artifact = self.store.read_record(reference).record
            except EvidenceError:
                raise ApplicationServiceError("cleanup_artifact_missing") from None
            if not isinstance(artifact, Artifact):
                raise ApplicationServiceError("cleanup_artifact_invalid")
            values.append((reference, artifact))
        return tuple(values)

    def _append_object_removed(self, intent: _CleanupIntent, entry_id: str) -> None:
        item = next(
            (value for value in intent.objects if value.entry_id == entry_id), None
        )
        if item is None:
            raise ApplicationServiceError("cleanup_intent_invalid")
        self._append_cleanup(
            intent.stream_id,
            f"{intent.execution_id}-removed-{entry_id}",
            "cleanup_object_removed",
            {
                "entry_id": entry_id,
                "byte_class": item.byte_class.value,
                "byte_count": item.byte_count,
            },
        )

    def _record_failure(
        self,
        intent: _CleanupIntent,
        failure_code: str,
        entry_id: str | None = None,
    ) -> None:
        try:
            suffix = entry_id if entry_id is not None else "execution"
            self._append_cleanup(
                intent.stream_id,
                f"{intent.execution_id}-failure-{suffix}-{failure_code}",
                "cleanup_failure_observed",
                {
                    "entry_id": entry_id,
                    "failure_code": failure_code,
                },
            )
        except ApplicationServiceError:
            pass

    def _append_terminal(self, intent: _CleanupIntent, receipt: CleanupReceipt) -> None:
        self._append_cleanup(
            intent.stream_id,
            f"{intent.execution_id}-terminal",
            f"cleanup_{receipt.outcome.value}",
            {
                "receipt_identity": identity_fields(receipt.identity),
                "logical_bytes_removed": receipt.logical_bytes_removed,
                "physical_bytes_freed": receipt.physical_bytes_freed,
                "failure_code": receipt.failure_code,
            },
        )

    def _receipt_by_id(self, receipt_id: str) -> CleanupReceipt | None:
        values = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, CleanupReceipt)
            and stored.record.receipt_id == receipt_id
        )
        if len(values) > 1:
            raise ApplicationServiceError("cleanup_receipt_ambiguous")
        return values[0] if values else None

    def _availability_by_id(self, availability_id: str) -> ArtifactAvailability | None:
        values = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, ArtifactAvailability)
            and stored.record.availability_id == availability_id
        )
        if len(values) > 1:
            raise ApplicationServiceError("artifact_availability_ambiguous")
        return values[0] if values else None

    def _current_availability(self, artifact: Artifact) -> ArtifactAvailability:
        values = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, ArtifactAvailability)
            and stored.record.artifact == record_reference(artifact)
        )
        superseded = {
            value.supersedes.identity
            for value in values
            if value.supersedes is not None
        }
        current = tuple(value for value in values if value.identity not in superseded)
        if len(current) != 1:
            raise ApplicationServiceError("artifact_availability_ambiguous")
        return current[0]

    def _reconcile_pending(self) -> None:
        try:
            self.store.verify()
            snapshots = self.store.iter_streams()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        if not self._cleanup_lock_held and any(
            event.event_type == "cleanup_started"
            for snapshot in snapshots
            for event in snapshot.events
        ):
            with self._claim_cleanup_execution():
                self._reconcile_pending()
            return
        for snapshot in snapshots:
            started = next(
                (
                    event
                    for event in snapshot.events
                    if event.event_type == "cleanup_started"
                ),
                None,
            )
            if started is None:
                continue
            try:
                intent = _intent_from_payload(started.payload)
            except (RecordValidationError, ValueError, TypeError):
                raise ApplicationServiceError("cleanup_intent_invalid") from None
            if snapshot.stream_id != intent.stream_id:
                raise ApplicationServiceError("cleanup_intent_invalid")
            with self._claim_producer_ownership(intent.objects):
                self._reconcile_intent(intent, snapshot.events)

    def _reconcile_intent(
        self,
        intent: _CleanupIntent,
        events: tuple[Any, ...],
    ) -> None:
        existing = self._receipt_by_id(intent.receipt_id)
        if existing is not None:
            if (
                existing.execution_id != intent.execution_id
                or existing.plan_identity != intent.plan_identity
                or existing.inventory_identity != intent.inventory_identity
            ):
                raise ApplicationServiceError("cleanup_execution_conflict")
            statuses = {item.entry_id: item.status for item in existing.objects}
            _, safety_failure = self._settle_safety_fences(intent, statuses)
            if safety_failure is not None:
                self._record_failure(intent, safety_failure)
            try:
                self._append_terminal(intent, existing)
            except ApplicationServiceError:
                pass
            return
        statuses = self._reconciled_statuses(intent, events)
        failure_code = next(
            (
                value
                for event in reversed(events)
                if event.event_type == "cleanup_failure_observed"
                and isinstance((value := event.payload.get("failure_code")), str)
                and value
            ),
            "cleanup_reconciled_after_interruption",
        )
        self._finalize_execution(intent, statuses, failure_code)

    def _reconciled_statuses(
        self,
        intent: _CleanupIntent,
        events: tuple[Any, ...],
    ) -> dict[str, CleanupObjectStatus]:
        deletion_intents = {
            entry_id
            for event in events
            if event.event_type == "cleanup_object_deletion_intent"
            and isinstance((entry_id := event.payload.get("entry_id")), str)
        }
        recorded_removed = {
            entry_id
            for event in events
            if event.event_type == "cleanup_object_removed"
            and isinstance((entry_id := event.payload.get("entry_id")), str)
        }
        definitely_retained = {
            entry_id
            for event in events
            if event.event_type == "cleanup_failure_observed"
            and event.payload.get("failure_code") == "cleanup_object_remove_failed"
            and isinstance((entry_id := event.payload.get("entry_id")), str)
        }
        changed = {
            entry_id
            for event in events
            if event.event_type == "cleanup_failure_observed"
            and event.payload.get("failure_code") == "cleanup_object_changed"
            and isinstance((entry_id := event.payload.get("entry_id")), str)
        }
        known_ids = set(intent.selected_entry_ids)
        if (
            not deletion_intents <= known_ids
            or not recorded_removed <= known_ids
            or not definitely_retained <= known_ids
            or not changed <= known_ids
        ):
            raise ApplicationServiceError("cleanup_intent_invalid")
        statuses: dict[str, CleanupObjectStatus] = {}
        for item in intent.objects:
            if item.entry_id in recorded_removed:
                statuses[item.entry_id] = CleanupObjectStatus.REMOVED
                continue
            if item.entry_id in changed:
                statuses[item.entry_id] = CleanupObjectStatus.AMBIGUOUS
                continue
            if item.entry_id not in deletion_intents:
                statuses[item.entry_id] = CleanupObjectStatus.NOT_ATTEMPTED
                continue
            path = self._runtime_root().joinpath(*PurePosixPath(item.logical_key).parts)
            try:
                info = safe_path_stat(path, allow_missing=True)
                if info is None:
                    statuses[item.entry_id] = CleanupObjectStatus.REMOVED
                else:
                    snapshot = _stream_file_snapshot(
                        path, chunk_size=self._hash_chunk_size
                    )
                    statuses[item.entry_id] = (
                        CleanupObjectStatus.RETAINED
                        if item.entry_id in definitely_retained
                        and snapshot.byte_count == item.byte_count
                        and snapshot.content_identity == item.content_identity
                        else CleanupObjectStatus.AMBIGUOUS
                    )
            except (OSError, SafeIoError, UnsafeFilesystemPath):
                raise ApplicationServiceError("cleanup_reconciliation_unsafe") from None
        return statuses

    def _append_cleanup(
        self,
        stream_id: str,
        event_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        try:
            self.store.append_event(
                stream_id, EventRequest(event_id, event_type, payload)
            )
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None

    def _released_run_ids(self, records: tuple[TypedRecord, ...]) -> frozenset[str]:
        released: set[str] = set()
        root = self._runtime_root()
        for record in records:
            if not isinstance(record, Run):
                continue
            try:
                released_run_claim_identity(root, record.run_id)
            except RunOwnershipError:
                continue
            released.add(record.run_id)
        return frozenset(released)

    @contextmanager
    def _claim_producer_ownership(self, objects: Iterable[Any]) -> Iterator[None]:
        run_ids = self._producer_run_ids(objects)
        claims: list[tuple[str, ContentIdentity]] = []
        try:
            for run_id in run_ids:
                claims.append(
                    (
                        run_id,
                        released_run_claim_identity(self._runtime_root(), run_id),
                    )
                )
            with ExitStack() as stack:
                for run_id, claim_identity in claims:
                    stack.enter_context(
                        claim_released_run_ownership(
                            self._runtime_root(), run_id, claim_identity
                        )
                    )
                yield
        except RunOwnershipError:
            raise ApplicationServiceError(
                "cleanup_producer_ownership_unavailable"
            ) from None

    def _producer_run_ids(self, objects: Iterable[Any]) -> tuple[str, ...]:
        run_ids: set[str] = set()

        def add_run(reference: RecordReference) -> None:
            run = self._read_exact_record(
                reference, Run, "cleanup_producer_evidence_invalid"
            )
            run_ids.add(run.run_id)

        for item in objects:
            byte_class = getattr(item, "byte_class", None)
            subjects = getattr(item, "subjects", ())
            if not isinstance(subjects, tuple):
                raise ApplicationServiceError("cleanup_producer_evidence_invalid")
            if byte_class in {ByteClass.CHECKPOINT, ByteClass.STAGING_CACHE}:
                for subject in subjects:
                    if subject.record_type == "run":
                        add_run(subject)
            elif byte_class is ByteClass.FINAL_ADAPTER:
                for subject in subjects:
                    if subject.record_type != "artifact":
                        continue
                    artifact = self._read_exact_record(
                        subject, Artifact, "cleanup_producer_evidence_invalid"
                    )
                    add_run(artifact.producing_run)
            elif byte_class is ByteClass.EXPORT_BUNDLE:
                for subject in subjects:
                    if subject.record_type != "adapter_export":
                        continue
                    exported = self._read_exact_record(
                        subject, AdapterExport, "cleanup_producer_evidence_invalid"
                    )
                    artifact = self._read_exact_record(
                        exported.artifact,
                        Artifact,
                        "cleanup_producer_evidence_invalid",
                    )
                    add_run(artifact.producing_run)
        return tuple(sorted(run_ids))

    def _read_exact_record(
        self,
        reference: RecordReference,
        kind: type[Any],
        code: str,
    ) -> Any:
        try:
            record = self.store.read_record(reference).record
        except EvidenceError:
            raise ApplicationServiceError(code) from None
        if not isinstance(record, kind):
            raise ApplicationServiceError(code)
        return record

    def _record_by_identity(
        self,
        kind: type[Any],
        identity: ContentIdentity,
        code: str,
    ) -> Any:
        try:
            matches = tuple(
                stored.record
                for stored in self.store.iter_records()
                if isinstance(stored.record, kind)
                and stored.record.identity == identity
            )
        except EvidenceError:
            raise ApplicationServiceError(code) from None
        if len(matches) != 1:
            raise ApplicationServiceError(code)
        return matches[0]

    def _project(self) -> Project:
        projects = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, Project)
        )
        if len(projects) != 1:
            raise ApplicationServiceError("cleanup_project_ambiguous")
        return projects[0]

    def _runtime_root(self) -> Path:
        return Path(os.path.abspath(self.project_root / RUNTIME_OUTPUT_DIRECTORY))


def _lock_cleanup_handle(handle: Any) -> None:
    try:
        if os.name == "nt":
            msvcrt: Any = importlib.import_module("msvcrt")
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        deadlock = getattr(errno, "EDEADLK", None)
        if (
            isinstance(exc, BlockingIOError)
            or exc.errno in {errno.EACCES, errno.EAGAIN}
            or (deadlock is not None and exc.errno == deadlock)
        ):
            raise ApplicationServiceError("cleanup_execution_busy") from None
        raise ApplicationServiceError("cleanup_lock_unavailable") from None


def _unlock_cleanup_handle(handle: Any) -> None:
    if os.name == "nt":
        msvcrt: Any = importlib.import_module("msvcrt")
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl: Any = importlib.import_module("fcntl")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _receipt_id(execution_id: str) -> str:
    if _EXECUTION_ID.fullmatch(execution_id) is None:
        raise ApplicationServiceError("cleanup_execution_id_invalid")
    return f"cleanup-receipt-{execution_id.removeprefix('cleanup-execution-')}"


def _availability_id(
    disposition: str,
    intent: _CleanupIntent,
    artifact: Artifact,
) -> str:
    return f"cleanup-{disposition}-{intent.token}-{artifact.identity.value[:16]}"


def _intent_from_payload(value: Mapping[str, Any]) -> _CleanupIntent:
    required = {
        "intent_schema_version",
        "execution_id",
        "receipt_id",
        "project",
        "inventory_identity",
        "plan_identity",
        "selected_entry_ids",
        "objects",
        "physical_group_ids_freed",
        "impact_categories",
        "affected_subjects",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecordValidationError("cleanup intent fields are invalid")
    if value["intent_schema_version"] != "v1":
        raise RecordValidationError("cleanup intent schema is invalid")
    execution_id = value["execution_id"]
    receipt_id = value["receipt_id"]
    if (
        not isinstance(execution_id, str)
        or _EXECUTION_ID.fullmatch(execution_id) is None
        or not isinstance(receipt_id, str)
        or receipt_id != _receipt_id(execution_id)
    ):
        raise RecordValidationError("cleanup intent execution is invalid")
    require_identifier("receipt_id", receipt_id)
    project = _reference_from_payload(value["project"], "cleanup intent project")
    if project.record_type != "project":
        raise RecordValidationError("cleanup intent project is invalid")
    inventory_identity = parse_identity(
        _mapping(value["inventory_identity"], "cleanup intent inventory"),
        field="cleanup intent inventory",
    )
    plan_identity = parse_identity(
        _mapping(value["plan_identity"], "cleanup intent plan"),
        field="cleanup intent plan",
    )
    selected = _string_tuple(
        value["selected_entry_ids"],
        "cleanup intent selected entries",
        non_empty=True,
    )
    raw_objects = value["objects"]
    if not isinstance(raw_objects, (list, tuple)) or not raw_objects:
        raise RecordValidationError("cleanup intent objects are invalid")
    objects = tuple(_intent_object_from_payload(item) for item in raw_objects)
    if tuple(item.entry_id for item in objects) != selected:
        raise RecordValidationError("cleanup intent object order is invalid")
    freed_groups = _string_tuple(
        value["physical_group_ids_freed"],
        "cleanup intent physical groups",
        non_empty=False,
    )
    object_groups = {item.physical_group_id for item in objects}
    if not set(freed_groups) <= object_groups:
        raise RecordValidationError("cleanup intent physical groups are invalid")
    impacts = _impact_tuple(value["impact_categories"], "cleanup intent impacts")
    object_impacts = {impact for item in objects for impact in item.impacts}
    if not object_impacts <= set(impacts):
        raise RecordValidationError("cleanup intent impacts are incomplete")
    affected = _reference_tuple(
        value["affected_subjects"], "cleanup intent affected subjects"
    )
    if affected != _sorted_references(
        subject for item in objects for subject in item.subjects
    ):
        raise RecordValidationError("cleanup intent affected subjects are invalid")
    return _CleanupIntent(
        execution_id=execution_id,
        receipt_id=receipt_id,
        project=project,
        inventory_identity=inventory_identity,
        plan_identity=plan_identity,
        selected_entry_ids=selected,
        objects=objects,
        physical_group_ids_freed=freed_groups,
        impact_categories=impacts,
        affected_subjects=affected,
    )


def _intent_object_from_payload(value: Any) -> _CleanupIntentObject:
    required = {
        "entry_id",
        "logical_key",
        "byte_class",
        "byte_count",
        "content_identity",
        "physical_group_id",
        "subjects",
        "impacts",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecordValidationError("cleanup intent object fields are invalid")
    entry_id = value["entry_id"]
    logical_key = value["logical_key"]
    byte_class_value = value["byte_class"]
    byte_count = value["byte_count"]
    physical_group_id = value["physical_group_id"]
    if (
        not isinstance(entry_id, str)
        or not isinstance(logical_key, str)
        or not isinstance(byte_class_value, str)
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or not isinstance(physical_group_id, str)
    ):
        raise RecordValidationError("cleanup intent object values are invalid")
    require_identifier("entry_id", entry_id)
    require_identifier("physical_group_id", physical_group_id)
    byte_class = ByteClass(byte_class_value)
    identity = parse_identity(
        _mapping(value["content_identity"], "cleanup intent content identity"),
        field="cleanup intent content identity",
    )
    subjects = _reference_tuple(value["subjects"], "cleanup intent subjects")
    impacts = _impact_tuple(value["impacts"], "cleanup intent object impacts")
    CleanupObjectReceipt(
        entry_id=entry_id,
        logical_key=logical_key,
        byte_class=byte_class.value,
        byte_count=byte_count,
        content_identity=identity,
        status=CleanupObjectStatus.NOT_ATTEMPTED,
        physical_bytes_freed=False,
        subjects=subjects,
    )
    return _CleanupIntentObject(
        entry_id=entry_id,
        logical_key=logical_key,
        byte_class=byte_class,
        byte_count=byte_count,
        content_identity=identity,
        physical_group_id=physical_group_id,
        subjects=subjects,
        impacts=impacts,
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{field} is invalid")
    return value


def _string_tuple(value: Any, field: str, *, non_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RecordValidationError(f"{field} is invalid")
    result = tuple(value)
    if (non_empty and not result) or tuple(sorted(set(result))) != result:
        raise RecordValidationError(f"{field} is invalid")
    for item in result:
        require_identifier(field, item)
    return result


def _reference_from_payload(value: Any, field: str) -> RecordReference:
    if not isinstance(value, Mapping) or set(value) != {
        "record_kind",
        "record_id",
        "record_identity",
    }:
        raise RecordValidationError(f"{field} is invalid")
    record_type = value["record_kind"]
    logical_id = value["record_id"]
    if not isinstance(record_type, str) or not isinstance(logical_id, str):
        raise RecordValidationError(f"{field} is invalid")
    return RecordReference(
        record_type,
        logical_id,
        parse_identity(
            _mapping(value["record_identity"], f"{field} identity"),
            field=f"{field} identity",
        ),
    )


def _event_reference(value: RecordReference) -> dict[str, object]:
    """Encode a reference without using the reserved canonical wire shape."""

    return {
        "record_kind": value.record_type,
        "record_id": value.logical_id,
        "record_identity": identity_fields(value.identity),
    }


def _reference_tuple(value: Any, field: str) -> tuple[RecordReference, ...]:
    if not isinstance(value, (list, tuple)):
        raise RecordValidationError(f"{field} is invalid")
    references = tuple(_reference_from_payload(item, f"{field} item") for item in value)
    if references != _sorted_references(references):
        raise RecordValidationError(f"{field} is invalid")
    return references


def _impact_tuple(value: Any, field: str) -> tuple[CleanupImpact, ...]:
    if not isinstance(value, (list, tuple)):
        raise RecordValidationError(f"{field} is invalid")
    if any(not isinstance(item, str) for item in value):
        raise RecordValidationError(f"{field} is invalid")
    result = tuple(CleanupImpact(item) for item in value)
    if tuple(sorted(set(result), key=lambda item: item.value)) != result:
        raise RecordValidationError(f"{field} is invalid")
    return result


def _enumerate_regular_files(root: Path) -> tuple[Path, ...]:
    pending = [root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        require_safe_directory(directory)
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda item: item.name)
        for entry in ordered:
            info = entry.stat(follow_symlinks=False)
            if is_link_or_reparse(info):
                raise ApplicationServiceError("storage_inventory_link_forbidden")
            path = Path(entry.path)
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                files.append(path)
            else:
                raise ApplicationServiceError("storage_inventory_type_forbidden")
    return tuple(sorted(files, key=lambda path: path.as_posix()))


def _classify(
    logical_key: str,
    records: tuple[TypedRecord, ...],
    streams: tuple[Any, ...],
    manifests: tuple[BundleManifest, ...],
    observed_identity: ContentIdentity,
    released_run_ids: frozenset[str],
) -> tuple[
    ByteClass,
    tuple[RecordReference, ...],
    tuple[CleanupImpact, ...],
    bool,
]:
    parts = PurePosixPath(logical_key).parts
    if len(parts) >= 3 and parts[0] == "artifacts":
        subjects = _records_by_id(records, Artifact, parts[1])
        relative = PurePosixPath(*parts[2:]).as_posix()
        artifacts = tuple(
            record
            for record in records
            if isinstance(record, Artifact) and record.artifact_id == parts[1]
        )
        known_member = len(artifacts) == 1 and any(
            manifest.identity == artifacts[0].content_identity
            and any(member.path == relative for member in manifest.members)
            for manifest in manifests
        )
        if not subjects or not known_member or len(artifacts) != 1:
            return _protected_unknown(subjects)
        producer_settled = _run_output_settled(
            artifacts[0].producing_run.logical_id,
            streams,
            released_run_ids,
        )
        return (
            ByteClass.FINAL_ADAPTER,
            subjects,
            (
                CleanupImpact.FINAL_ARTIFACT_AVAILABILITY,
                CleanupImpact.INSPECTABILITY,
            ),
            bool(subjects) and producer_settled,
        )
    if len(parts) >= 3 and parts[0] == "checkpoints":
        subjects = _records_by_id(records, Run, parts[1])
        runs = tuple(
            record
            for record in records
            if isinstance(record, Run) and record.run_id == parts[1]
        )
        checkpoint_identity = _checkpoint_identity(parts[1], logical_key, streams)
        if (
            not subjects
            or len(runs) != 1
            or checkpoint_identity is None
            or checkpoint_identity != observed_identity
        ):
            return _protected_unknown(subjects)
        impacts = {CleanupImpact.INSPECTABILITY, CleanupImpact.DEBUGGING_EVIDENCE}
        if _checkpoint_resumable(parts[1], logical_key, streams):
            impacts.add(CleanupImpact.RESUMABILITY)
        return (
            ByteClass.CHECKPOINT,
            subjects,
            tuple(sorted(impacts, key=lambda item: item.value)),
            bool(subjects) and _run_output_settled(parts[1], streams, released_run_ids),
        )
    if len(parts) >= 3 and parts[0] == "exports":
        subjects = _records_by_id(records, AdapterExport, parts[1])
        exports = tuple(
            record
            for record in records
            if isinstance(record, AdapterExport) and record.export_id == parts[1]
        )
        relative = PurePosixPath(*parts[2:]).as_posix()
        known_members = {
            EXPORT_MANIFEST_MEMBER,
            *(
                f"{EXPORT_BUNDLE_PREFIX}/{member}"
                for member in FIXTURE_ARTIFACT_MEMBERS
            ),
        }
        if not subjects or len(exports) != 1 or relative not in known_members:
            return _protected_unknown(subjects)
        artifact = _record_for_reference(records, exports[0].artifact, Artifact)
        if artifact is None:
            return _protected_unknown(subjects)
        return (
            ByteClass.EXPORT_BUNDLE,
            subjects,
            (CleanupImpact.INSPECTABILITY,),
            bool(subjects)
            and _run_output_settled(
                artifact.producing_run.logical_id,
                streams,
                released_run_ids,
            ),
        )
    if parts and parts[0] == "runtime-ownership":
        subjects = _records_by_id(records, Run, parts[1]) if len(parts) > 1 else ()
        return (
            ByteClass.RUNTIME_CONTROL,
            subjects,
            (CleanupImpact.INSPECTABILITY,),
            False,
        )
    if parts and parts[0] == "library-staging":
        subjects = _records_by_id(records, Run, parts[1]) if len(parts) > 1 else ()
        runs = tuple(
            record
            for record in records
            if isinstance(record, Run) and len(parts) > 1 and record.run_id == parts[1]
        )
        return (
            ByteClass.STAGING_CACHE,
            subjects,
            (CleanupImpact.CACHE_CONVENIENCE, CleanupImpact.DEBUGGING_EVIDENCE),
            bool(subjects)
            and len(runs) == 1
            and _run_output_settled(parts[1], streams, released_run_ids),
        )
    if parts and parts[0] == "logs":
        return (
            ByteClass.DEBUG_EVIDENCE,
            (),
            (CleanupImpact.DEBUGGING_EVIDENCE,),
            True,
        )
    return _protected_unknown(())


def _protected_unknown(
    subjects: tuple[RecordReference, ...],
) -> tuple[
    ByteClass,
    tuple[RecordReference, ...],
    tuple[CleanupImpact, ...],
    bool,
]:
    return ByteClass.UNKNOWN, subjects, (CleanupImpact.INSPECTABILITY,), False


def _records_by_id(
    records: tuple[TypedRecord, ...], kind: type[TypedRecord], logical_id: str
) -> tuple[RecordReference, ...]:
    matches = tuple(
        record_reference(record)
        for record in records
        if isinstance(record, kind)
        and getattr(record, _logical_id_field(record), None) == logical_id
    )
    return _sorted_references(matches)


def _record_for_reference(
    records: tuple[TypedRecord, ...],
    reference: RecordReference,
    kind: type[Any],
) -> Any | None:
    matches = tuple(
        record
        for record in records
        if isinstance(record, kind) and record_reference(record) == reference
    )
    return matches[0] if len(matches) == 1 else None


def _logical_id_field(record: TypedRecord) -> str:
    return {
        "artifact": "artifact_id",
        "adapter_export": "export_id",
        "run": "run_id",
    }.get(record.RECORD_TYPE, "")


def _checkpoint_identity(
    run_id: str,
    logical_key: str,
    streams: tuple[Any, ...],
) -> ContentIdentity | None:
    name = PurePosixPath(logical_key).name
    for snapshot in streams:
        if getattr(snapshot, "stream_id", None) != f"run-{run_id}":
            continue
        for event in snapshot.events:
            if event.event_type != "run_checkpoint":
                continue
            step = event.payload.get("step")
            identity = event.payload.get("checkpoint_identity")
            if (
                isinstance(step, int)
                and not isinstance(step, bool)
                and step >= 0
                and isinstance(identity, Mapping)
                and isinstance(identity.get("value"), str)
                and name == f"{step:08d}-{identity['value']}.json"
            ):
                try:
                    return parse_identity(identity, field="checkpoint identity")
                except RecordValidationError:
                    return None
        return None
    return None


def _checkpoint_resumable(
    run_id: str, logical_key: str, streams: tuple[Any, ...]
) -> bool:
    name = PurePosixPath(logical_key).name
    for snapshot in streams:
        if getattr(snapshot, "stream_id", None) != f"run-{run_id}":
            continue
        resumable = False
        pending: set[tuple[str, str]] = set()
        removed = False
        for event in snapshot.events:
            if event.event_type in {
                "run_checkpoint_cleanup_pending",
                "run_checkpoint_cleanup_cancelled",
                "run_checkpoint_removed",
            }:
                observation = event.payload.get("content_identity")
                if (
                    isinstance(observation, Mapping)
                    and isinstance(observation.get("value"), str)
                    and observation["value"] in name
                ):
                    execution_id = event.payload.get("execution_id")
                    entry_id = event.payload.get("entry_id")
                    if not isinstance(execution_id, str) or not isinstance(
                        entry_id, str
                    ):
                        return False
                    key = (execution_id, entry_id)
                    if event.event_type == "run_checkpoint_cleanup_pending":
                        pending.add(key)
                    elif key not in pending:
                        return False
                    else:
                        pending.discard(key)
                        removed = (
                            removed or event.event_type == "run_checkpoint_removed"
                        )
                continue
            if event.event_type != "run_checkpoint":
                continue
            identity = event.payload.get("checkpoint_identity")
            if (
                isinstance(identity, Mapping)
                and isinstance(identity.get("value"), str)
                and identity["value"] in name
            ):
                resumable = event.payload.get("resume_compatible") is True
        return resumable and not removed and not pending
    return False


def _run_output_settled(
    run_id: str,
    streams: tuple[Any, ...],
    released_run_ids: frozenset[str],
) -> bool:
    if run_id not in released_run_ids:
        return False
    matches = tuple(
        snapshot for snapshot in streams if snapshot.stream_id == f"run-{run_id}"
    )
    if len(matches) != 1:
        return False
    try:
        status, lifecycle = validated_run_lifecycle(matches[0].events)
    except ApplicationServiceError:
        return False
    return (
        status.terminal
        and sum(event.event_type == "run_launched" for event in lifecycle) == 1
    )


def _sorted_references(
    values: Iterable[RecordReference],
) -> tuple[RecordReference, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda item: (item.record_type, item.logical_id, item.identity.value),
        )
    )


def _stream_file_snapshot(
    path: Path,
    *,
    chunk_size: int = _HASH_CHUNK_BYTES,
) -> _StableFileSnapshot:
    """Hash one stable regular-file snapshot without loading heavy bytes at once."""

    if chunk_size <= 0:
        raise SafeIoError("storage snapshot chunk size is invalid")
    try:
        before = require_safe_regular_file(path)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        digest = hashlib.sha256()
        bytes_read = 0
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                is_link_or_reparse(opened)
                or not stat.S_ISREG(opened.st_mode)
                or not same_file_object(before, opened)
            ):
                raise SafeIoError("storage file changed while opening")
            while chunk := handle.read(chunk_size):
                bytes_read += len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        current = require_safe_regular_file(path)
    except FileNotFoundError:
        raise
    except SafeIoError:
        raise
    except (OSError, UnsafeFilesystemPath) as exc:
        raise SafeIoError("unable to hash a stable storage file") from exc
    if (
        _signature(opened) != _signature(after)
        or _signature(after) != _signature(current)
        or not same_file_object(after, current)
        or bytes_read != after.st_size
    ):
        raise SafeIoError("storage file changed while hashing")
    return _StableFileSnapshot(
        ContentIdentity("sha256", digest.hexdigest()),
        bytes_read,
        _signature(after),
    )


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_nlink,
    )
