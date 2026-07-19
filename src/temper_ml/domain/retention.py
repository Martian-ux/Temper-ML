"""Immutable cleanup receipts for explicit local heavy-byte removal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import ClassVar

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    identity_fields,
    require_identifier,
    require_non_negative_int,
    require_string_tuple,
)


_BYTE_CLASSES = frozenset(
    {
        "checkpoint",
        "debugging_evidence",
        "export_bundle",
        "final_adapter",
        "runtime_control",
        "staging_cache",
        "unknown",
    }
)
_IMPACT_CATEGORIES = frozenset(
    {
        "cache_convenience",
        "debugging_evidence",
        "final_artifact_availability",
        "inspectability",
        "resumability",
        "shared_reference",
    }
)


class CleanupOutcome(str, Enum):
    """Terminal disposition of one exact cleanup plan execution."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class CleanupObjectStatus(str, Enum):
    """Disposition of one selected logical storage entry."""

    REMOVED = "removed"
    RETAINED = "retained"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class CleanupObjectReceipt:
    """Portable result for one selected file, without a host path."""

    entry_id: str
    logical_key: str
    byte_class: str
    byte_count: int
    content_identity: ContentIdentity
    status: CleanupObjectStatus
    physical_bytes_freed: bool
    subjects: tuple[RecordReference, ...]

    def __post_init__(self) -> None:
        require_identifier("entry_id", self.entry_id)
        require_cleanup_logical_key(self.logical_key)
        require_identifier("byte_class", self.byte_class)
        if self.byte_class not in _BYTE_CLASSES:
            raise RecordValidationError("cleanup object byte_class is unknown")
        require_non_negative_int("byte_count", self.byte_count)
        if not isinstance(self.content_identity, ContentIdentity):
            raise RecordValidationError("content_identity must be a content identity")
        if not isinstance(self.status, CleanupObjectStatus):
            raise RecordValidationError("cleanup object status is invalid")
        if not isinstance(self.physical_bytes_freed, bool):
            raise RecordValidationError("physical_bytes_freed must be a boolean")
        if self.status is not CleanupObjectStatus.REMOVED and self.physical_bytes_freed:
            raise RecordValidationError(
                "an unremoved cleanup object cannot free physical bytes"
            )
        if not isinstance(self.subjects, tuple) or any(
            not isinstance(subject, RecordReference) for subject in self.subjects
        ):
            raise RecordValidationError("cleanup object subjects must be references")
        ordered = tuple(
            sorted(
                self.subjects,
                key=lambda item: (
                    item.record_type,
                    item.logical_id,
                    item.identity.value,
                ),
            )
        )
        if len(set(ordered)) != len(ordered):
            raise RecordValidationError("cleanup object subjects contain duplicates")
        object.__setattr__(self, "subjects", ordered)

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "logical_key": self.logical_key,
            "byte_class": self.byte_class,
            "byte_count": self.byte_count,
            "content_identity": identity_fields(self.content_identity),
            "status": self.status.value,
            "physical_bytes_freed": self.physical_bytes_freed,
            "subjects": [subject.to_dict() for subject in self.subjects],
        }


@dataclass(frozen=True)
class CleanupReceipt(TypedRecord):
    """Immutable outcome for one confirmed, snapshot-bound cleanup plan."""

    RECORD_TYPE: ClassVar[str] = "cleanup_receipt"

    receipt_id: str
    execution_id: str
    project: RecordReference
    inventory_identity: ContentIdentity
    plan_identity: ContentIdentity
    outcome: CleanupOutcome
    selected_entry_ids: tuple[str, ...]
    objects: tuple[CleanupObjectReceipt, ...]
    logical_bytes_removed: int
    physical_bytes_freed: int
    impact_categories: tuple[str, ...]
    affected_subjects: tuple[RecordReference, ...]
    availability_updates: tuple[RecordReference, ...]
    failure_code: str | None = None

    def __post_init__(self) -> None:
        require_identifier("receipt_id", self.receipt_id)
        require_identifier("execution_id", self.execution_id)
        if (
            not isinstance(self.project, RecordReference)
            or self.project.record_type != "project"
        ):
            raise RecordValidationError("project must reference a project")
        for field in ("inventory_identity", "plan_identity"):
            if not isinstance(getattr(self, field), ContentIdentity):
                raise RecordValidationError(f"{field} must be a content identity")
        if not isinstance(self.outcome, CleanupOutcome):
            raise RecordValidationError("cleanup outcome is invalid")
        selected = require_string_tuple(
            "selected_entry_ids",
            self.selected_entry_ids,
            sorted_values=True,
        )
        for entry_id in selected:
            require_identifier("selected_entry_id", entry_id)
        object.__setattr__(self, "selected_entry_ids", selected)
        if not isinstance(self.objects, tuple) or any(
            not isinstance(item, CleanupObjectReceipt) for item in self.objects
        ):
            raise RecordValidationError("cleanup objects must be a tuple")
        if tuple(item.entry_id for item in self.objects) != selected:
            raise RecordValidationError(
                "cleanup objects must exactly cover selected entries in order"
            )
        require_non_negative_int("logical_bytes_removed", self.logical_bytes_removed)
        require_non_negative_int("physical_bytes_freed", self.physical_bytes_freed)
        removed = tuple(
            item for item in self.objects if item.status is CleanupObjectStatus.REMOVED
        )
        if self.logical_bytes_removed != sum(item.byte_count for item in removed):
            raise RecordValidationError("logical removed bytes do not match objects")
        if self.physical_bytes_freed != sum(
            item.byte_count for item in removed if item.physical_bytes_freed
        ):
            raise RecordValidationError("physical freed bytes do not match objects")
        if self.physical_bytes_freed > self.logical_bytes_removed:
            raise RecordValidationError("physical freed bytes exceed logical removal")
        object.__setattr__(
            self,
            "impact_categories",
            require_string_tuple(
                "impact_categories",
                self.impact_categories,
                non_empty=False,
                sorted_values=True,
            ),
        )
        if any(item not in _IMPACT_CATEGORIES for item in self.impact_categories):
            raise RecordValidationError("cleanup impact category is unknown")
        object.__setattr__(
            self,
            "affected_subjects",
            _validated_references("affected_subjects", self.affected_subjects),
        )
        object.__setattr__(
            self,
            "availability_updates",
            _validated_references(
                "availability_updates",
                self.availability_updates,
                record_type="artifact_availability",
            ),
        )
        if self.failure_code is not None:
            require_identifier("failure_code", self.failure_code)
        statuses = {item.status for item in self.objects}
        if self.outcome is CleanupOutcome.COMPLETED:
            if (
                statuses != {CleanupObjectStatus.REMOVED}
                or self.failure_code is not None
            ):
                raise RecordValidationError(
                    "completed cleanup requires every object removed and no failure"
                )
        elif self.outcome is CleanupOutcome.FAILED:
            if removed or self.failure_code is None:
                raise RecordValidationError(
                    "failed cleanup requires no removals and a failure code"
                )
        elif not removed or self.failure_code is None:
            raise RecordValidationError(
                "partial cleanup requires some removals and a failure code"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "execution_id": self.execution_id,
            "project": self.project.to_dict(),
            "inventory_identity": identity_fields(self.inventory_identity),
            "plan_identity": identity_fields(self.plan_identity),
            "outcome": self.outcome.value,
            "selected_entry_ids": list(self.selected_entry_ids),
            "objects": [item.to_dict() for item in self.objects],
            "logical_bytes_removed": self.logical_bytes_removed,
            "physical_bytes_freed": self.physical_bytes_freed,
            "impact_categories": list(self.impact_categories),
            "affected_subjects": [
                subject.to_dict() for subject in self.affected_subjects
            ],
            "availability_updates": [
                update.to_dict() for update in self.availability_updates
            ],
            "failure_code": self.failure_code,
        }


def require_cleanup_logical_key(value: str) -> str:
    """Require the portable relative key shared by cleanup intent and receipt."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise RecordValidationError("logical_key must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path == "."
        or ".." in path.parts
        or (len(value) >= 2 and value[1] == ":")
    ):
        raise RecordValidationError("logical_key must be a portable relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RecordValidationError("logical_key contains an unsafe component")
    return value


def _validated_references(
    field: str,
    value: tuple[RecordReference, ...],
    *,
    record_type: str | None = None,
) -> tuple[RecordReference, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, RecordReference)
        or (record_type is not None and item.record_type != record_type)
        for item in value
    ):
        raise RecordValidationError(f"{field} must contain valid references")
    ordered = tuple(
        sorted(
            value,
            key=lambda item: (item.record_type, item.logical_id, item.identity.value),
        )
    )
    if len(set(ordered)) != len(ordered):
        raise RecordValidationError(f"{field} contains duplicate references")
    return ordered
