"""Typed canonical evidence services over the project-local Temper store."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from temper_ml.domain.artifacts import ArtifactError, BundleManifest
from temper_ml.domain.projections import ContentIdentity, ProjectionError
from temper_ml.domain.records import (
    CORE_LOGICAL_ID_FIELDS,
    CORE_PROJECTION_REGISTRY,
    RecordEnvelope,
    RecordReference,
    RecordValidationError,
    TypedRecord,
    thaw_json,
)
from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    require_safe_directory,
    require_safe_regular_file,
    safe_path_stat,
)
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)
from temper_ml.store.event_stream import (
    EventConflict,
    EventRequest,
    EventStream,
    EventStreamCorrupt,
    StoredEvent,
)
from temper_ml.store.layout import StoreLayout, StorePathError
from temper_ml.store.redaction import (
    RedactionContext,
    RedactionResult,
    PublicSafetyError,
    public_export_wrapper,
    validate_canonical_admission,
)
from temper_ml.store.safe_io import (
    SafeIoError,
    read_stable_bytes,
    replace_bytes,
    write_once_bytes,
)
from temper_ml.store.verifier import (
    VerificationError,
    verify_bundle,
    verify_file,
)

_SHA256_FILE = re.compile(r"^([0-9a-f]{64})\.json$")
_TEMP_FILE = re.compile(r"^\.[0-9a-f]{64}\.json\.[0-9a-f]{32}\.tmp$")
_REFERENCE_WIRE_FIELDS = {"record_type", "logical_id", "identity"}
_ENVELOPE_WIRE_FIELDS = {
    "record_type",
    "schema_version",
    "projection_version",
    "identity",
    "payload",
}


class EvidenceError(RuntimeError):
    """Public-safe evidence error exposing only a stable symbolic code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EvidenceExists(EvidenceError):
    pass


class EvidenceCorrupt(EvidenceError):
    pass


class EvidenceNotFound(EvidenceError):
    pass


class EvidenceAmbiguous(EvidenceError):
    pass


@dataclass(frozen=True)
class StoredTypedRecord:
    envelope: RecordEnvelope
    record: TypedRecord
    path: Path

    @property
    def reference(self) -> RecordReference:
        logical_field = CORE_LOGICAL_ID_FIELDS[self.envelope.record_type]
        logical_id = getattr(self.record, logical_field)
        return RecordReference(
            self.envelope.record_type,
            logical_id,
            self.envelope.identity,
        )


@dataclass(frozen=True)
class StreamSnapshot:
    stream_id: str
    events: tuple[StoredEvent, ...]


@dataclass(frozen=True)
class ReconstructedProject:
    records: tuple[StoredTypedRecord, ...]
    streams: tuple[StreamSnapshot, ...]
    bundle_manifests: tuple[BundleManifest, ...]


@dataclass(frozen=True)
class ProjectVerification:
    record_counts: Mapping[str, int]
    record_count: int
    stream_count: int
    event_count: int
    bundle_manifest_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "record_counts",
            MappingProxyType(dict(sorted(self.record_counts.items()))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "verified",
            "record_count": self.record_count,
            "record_counts": dict(sorted(self.record_counts.items())),
            "event_stream_count": self.stream_count,
            "event_count": self.event_count,
            "bundle_manifest_count": self.bundle_manifest_count,
            "derived_state_rebuildable": True,
        }


class TypedEvidenceStore:
    """Canonical typed records, streams, manifests, verification, and dumps."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        redaction_context: RedactionContext | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.layout = StoreLayout(self.project_root)
        requested_context = (
            redaction_context if redaction_context is not None else RedactionContext()
        )
        self._admission_context = RedactionContext(
            local_usernames=requested_context.local_usernames,
            local_hostnames=requested_context.local_hostnames,
            allowed_public_url_prefixes=(),
        )
        self._portable_context = RedactionContext()

    def write_record(self, record: TypedRecord) -> StoredTypedRecord:
        if not isinstance(record, TypedRecord):
            raise EvidenceError("invalid_record_type")
        envelope = record.to_envelope()
        try:
            validate_canonical_admission(
                envelope.to_dict(), context=self._admission_context
            )
        except PublicSafetyError as exc:
            raise EvidenceError(f"admission_{exc.code}") from None
        path = self.layout.record_path(envelope.record_type, envelope.identity)
        existing = _safe_exists(path)
        if existing:
            stored = self._read_envelope_path(
                path, envelope.record_type, envelope.identity
            )
            if stored.envelope.to_dict() != envelope.to_dict():
                raise EvidenceCorrupt("record_identity_collision")
            raise EvidenceExists("record_exists")
        try:
            write_once_bytes(path, dumps_canonical_json(envelope.to_dict()))
        except FileExistsError:
            self._read_envelope_path(path, envelope.record_type, envelope.identity)
            raise EvidenceExists("record_exists") from None
        except SafeIoError as exc:
            raise EvidenceError("record_write_failed") from exc
        return self._read_envelope_path(path, envelope.record_type, envelope.identity)

    def read_record(self, reference: RecordReference) -> StoredTypedRecord:
        if not isinstance(reference, RecordReference):
            raise EvidenceError("invalid_record_reference")
        try:
            CORE_PROJECTION_REGISTRY.resolve(reference.record_type, "v1")
            path = self.layout.record_path(reference.record_type, reference.identity)
        except (ProjectionError, StorePathError) as exc:
            raise EvidenceError("invalid_record_reference") from exc
        if not _safe_exists(path):
            raise EvidenceNotFound("record_not_found")
        stored = self._read_envelope_path(
            path, reference.record_type, reference.identity
        )
        if stored.reference.logical_id != reference.logical_id:
            raise EvidenceCorrupt("record_logical_id_mismatch")
        return stored

    def iter_records(self) -> tuple[StoredTypedRecord, ...]:
        root = self.layout.records_root()
        if not _safe_exists(root):
            return ()
        known = {
            registration.record_type
            for registration in CORE_PROJECTION_REGISTRY.registrations
        }
        records: list[StoredTypedRecord] = []
        try:
            require_safe_directory(root)
            type_paths = sorted(root.iterdir(), key=lambda path: path.name)
            for type_path in type_paths:
                require_safe_directory(type_path)
                if type_path.name not in known:
                    raise EvidenceCorrupt("unknown_record_type_directory")
                entries = sorted(type_path.iterdir(), key=lambda path: path.name)
                if [entry.name for entry in entries] != ["sha256"]:
                    raise EvidenceCorrupt("invalid_record_algorithm_directory")
                algorithm_path = entries[0]
                require_safe_directory(algorithm_path)
                for path in sorted(
                    algorithm_path.iterdir(), key=lambda item: item.name
                ):
                    require_safe_regular_file(path)
                    if _TEMP_FILE.fullmatch(path.name):
                        continue
                    match = _SHA256_FILE.fullmatch(path.name)
                    if match is None:
                        raise EvidenceCorrupt("invalid_record_filename")
                    identity = ContentIdentity("sha256", match.group(1))
                    records.append(
                        self._read_envelope_path(path, type_path.name, identity)
                    )
        except EvidenceError:
            raise
        except (OSError, UnsafeFilesystemPath, ProjectionError) as exc:
            raise EvidenceCorrupt("unsafe_record_store") from exc
        return tuple(
            sorted(
                records,
                key=lambda stored: (
                    stored.envelope.record_type,
                    stored.envelope.identity.value,
                ),
            )
        )

    def append_event(self, stream_id: str, request: EventRequest) -> StoredEvent:
        try:
            path = self.layout.stream_events(stream_id)
            if _contains_untyped_evidence_shape(request.payload):
                raise EvidenceError("event_typed_evidence_unsupported")
            validate_canonical_admission(
                request.canonical_fields(), context=self._admission_context
            )
            return EventStream(path).append(request)
        except PublicSafetyError as exc:
            raise EvidenceError(f"admission_{exc.code}") from None
        except EventConflict as exc:
            raise EvidenceCorrupt("event_idempotency_conflict") from exc
        except (StorePathError, EventStreamCorrupt) as exc:
            raise EvidenceCorrupt("event_append_failed") from exc

    def iter_streams(self) -> tuple[StreamSnapshot, ...]:
        root = self.layout.streams_root()
        if not _safe_exists(root):
            return ()
        snapshots: list[StreamSnapshot] = []
        try:
            require_safe_directory(root)
            for stream_path in sorted(root.iterdir(), key=lambda path: path.name):
                require_safe_directory(stream_path)
                expected_events = self.layout.stream_events(stream_path.name)
                entries = sorted(stream_path.iterdir(), key=lambda path: path.name)
                if [entry.name for entry in entries] != ["events"]:
                    raise EvidenceCorrupt("invalid_stream_layout")
                require_safe_directory(expected_events)
                events = EventStream(expected_events).read_verified()
                for event in events:
                    if _contains_untyped_evidence_shape(event.payload):
                        raise EvidenceCorrupt("event_typed_evidence_unsupported")
                    validate_canonical_admission(
                        event.request_fields(), context=self._portable_context
                    )
                snapshots.append(StreamSnapshot(stream_path.name, events))
        except EvidenceError:
            raise
        except PublicSafetyError as exc:
            raise EvidenceCorrupt(f"event_{exc.code}") from None
        except (
            OSError,
            UnsafeFilesystemPath,
            StorePathError,
            EventStreamCorrupt,
        ) as exc:
            raise EvidenceCorrupt("unsafe_event_store") from exc
        return tuple(snapshots)

    def rebuild_stream_state(self, stream_id: str) -> Mapping[str, Any]:
        try:
            stream = EventStream(self.layout.stream_events(stream_id))
            state = stream.rebuild(_empty_stream_state(), _reduce_stream_state)
            validate_canonical_admission(state, context=self._portable_context)
            replace_bytes(
                self.layout.stream_state(stream_id), dumps_canonical_json(state)
            )
            return state
        except PublicSafetyError as exc:
            raise EvidenceCorrupt(f"derived_{exc.code}") from None
        except (StorePathError, EventStreamCorrupt, SafeIoError) as exc:
            raise EvidenceCorrupt("derived_state_rebuild_failed") from exc

    def rebuild_all_stream_states(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            self.rebuild_stream_state(snapshot.stream_id)
            for snapshot in self.iter_streams()
        )

    def write_bundle_manifest(self, manifest: BundleManifest) -> Path:
        if not isinstance(manifest, BundleManifest):
            raise EvidenceError("invalid_bundle_manifest")
        try:
            parsed = BundleManifest.from_dict(manifest.to_dict())
            validate_canonical_admission(
                parsed.to_dict(), context=self._admission_context
            )
            path = self.layout.bundle_manifest_path(parsed.identity)
            if _safe_exists(path):
                self._read_bundle_manifest_path(path, parsed.identity)
                raise EvidenceExists("bundle_manifest_exists")
            write_once_bytes(path, dumps_canonical_json(parsed.to_dict()))
            self._read_bundle_manifest_path(path, parsed.identity)
            return path
        except EvidenceError:
            raise
        except PublicSafetyError as exc:
            raise EvidenceError(f"admission_{exc.code}") from None
        except FileExistsError:
            self._read_bundle_manifest_path(path, parsed.identity)
            raise EvidenceExists("bundle_manifest_exists") from None
        except (ArtifactError, SafeIoError, StorePathError) as exc:
            raise EvidenceCorrupt("bundle_manifest_write_failed") from exc

    def read_bundle_manifest(self, identity: ContentIdentity) -> BundleManifest:
        try:
            path = self.layout.bundle_manifest_path(identity)
        except StorePathError as exc:
            raise EvidenceError("invalid_bundle_identity") from exc
        if not _safe_exists(path):
            raise EvidenceNotFound("bundle_manifest_not_found")
        return self._read_bundle_manifest_path(path, identity)

    def iter_bundle_manifests(self) -> tuple[BundleManifest, ...]:
        root = self.layout.bundle_manifests_root()
        parent = root.parent
        if not _safe_exists(parent):
            return ()
        manifests: list[BundleManifest] = []
        try:
            require_safe_directory(parent)
            entries = sorted(parent.iterdir(), key=lambda item: item.name)
            if [entry.name for entry in entries] != ["sha256"]:
                raise EvidenceCorrupt("invalid_bundle_algorithm_directory")
            require_safe_directory(root)
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                require_safe_regular_file(path)
                if _TEMP_FILE.fullmatch(path.name):
                    continue
                match = _SHA256_FILE.fullmatch(path.name)
                if match is None:
                    raise EvidenceCorrupt("invalid_bundle_manifest_filename")
                manifests.append(
                    self._read_bundle_manifest_path(
                        path, ContentIdentity("sha256", match.group(1))
                    )
                )
        except EvidenceError:
            raise
        except (OSError, UnsafeFilesystemPath, ProjectionError) as exc:
            raise EvidenceCorrupt("unsafe_bundle_manifest_store") from exc
        return tuple(sorted(manifests, key=lambda manifest: manifest.identity.value))

    def verify_file_evidence(self, path: Path | str, expected: ContentIdentity) -> None:
        candidate = self._project_path(path)
        try:
            verify_file(candidate, expected)
        except VerificationError as exc:
            raise EvidenceCorrupt("file_evidence_mismatch") from exc

    def verify_bundle_evidence(
        self, root: Path | str, manifest_identity: ContentIdentity
    ) -> None:
        candidate = self._project_path(root)
        manifest = self.read_bundle_manifest(manifest_identity)
        try:
            verify_bundle(candidate, manifest)
        except VerificationError as exc:
            raise EvidenceCorrupt("bundle_evidence_mismatch") from exc

    def reconstruct(self) -> ReconstructedProject:
        self._require_project_root()
        self._verify_store_surfaces()
        records = self.iter_records()
        self._verify_reference_closure(records)
        streams = self.iter_streams()
        manifests = self.iter_bundle_manifests()
        return ReconstructedProject(records, streams, manifests)

    def verify(self) -> ProjectVerification:
        reconstructed = self.reconstruct()
        return self._verification(reconstructed)

    def _verification(self, reconstructed: ReconstructedProject) -> ProjectVerification:
        counts = Counter(
            stored.envelope.record_type for stored in reconstructed.records
        )
        event_count = sum(len(snapshot.events) for snapshot in reconstructed.streams)
        for snapshot in reconstructed.streams:
            state: Mapping[str, Any] = _empty_stream_state()
            for event in snapshot.events:
                state = _reduce_stream_state(state, event)
            state_path = self.layout.stream_state(snapshot.stream_id)
            try:
                if _safe_exists(state_path):
                    require_safe_regular_file(state_path)
            except EvidenceError as exc:
                raise EvidenceCorrupt("derived_state_not_rebuildable") from exc
            except (OSError, UnsafeFilesystemPath) as exc:
                raise EvidenceCorrupt("derived_state_not_rebuildable") from exc
        return ProjectVerification(
            record_counts=dict(counts),
            record_count=len(reconstructed.records),
            stream_count=len(reconstructed.streams),
            event_count=event_count,
            bundle_manifest_count=len(reconstructed.bundle_manifests),
        )

    def public_dump(self) -> RedactionResult:
        reconstructed = self.reconstruct()
        verification = self._verification(reconstructed)
        records = (
            {
                "record_type": stored.envelope.record_type,
                "schema_version": stored.envelope.schema_version,
                "projection_version": stored.envelope.projection_version,
                "fields": thaw_json(stored.envelope.payload),
            }
            for stored in reconstructed.records
        )
        streams = (
            {
                "event_count": len(snapshot.events),
            }
            for snapshot in reconstructed.streams
        )
        return public_export_wrapper(
            records,
            record_counts=verification.record_counts,
            stream_summaries=streams,
            context=self._portable_context,
        )

    def inspect_manifest(
        self,
        record_type: str,
        logical_id: str,
        identity: ContentIdentity | None = None,
    ) -> RecordEnvelope:
        try:
            CORE_PROJECTION_REGISTRY.resolve(record_type, "v1")
        except ProjectionError as exc:
            raise EvidenceError("unknown_record_type") from exc
        records = self.iter_records()
        self._verify_reference_closure(records)
        matches = [
            stored
            for stored in records
            if stored.envelope.record_type == record_type
            and stored.reference.logical_id == logical_id
            and (identity is None or stored.envelope.identity == identity)
        ]
        if not matches:
            raise EvidenceNotFound("manifest_not_found")
        if len(matches) != 1:
            raise EvidenceAmbiguous("manifest_ambiguous")
        return matches[0].envelope

    def remove_record(self, reference: RecordReference) -> None:
        """Canonical records cannot be removed through the evidence service."""

        if not isinstance(reference, RecordReference):
            raise EvidenceError("invalid_record_reference")
        raise EvidenceError("immutable_cleanup_forbidden")

    def _read_envelope_path(
        self,
        path: Path,
        expected_type: str,
        expected_identity: ContentIdentity,
    ) -> StoredTypedRecord:
        try:
            raw = read_stable_bytes(path)
            value = loads_canonical_json(raw)
            if not isinstance(value, Mapping):
                raise EvidenceCorrupt("record_envelope_not_object")
            envelope = RecordEnvelope.from_dict(value)
            if raw != dumps_canonical_json(envelope.to_dict()):
                raise EvidenceCorrupt("record_not_canonical")
            if envelope.record_type != expected_type:
                raise EvidenceCorrupt("record_type_path_mismatch")
            if envelope.identity != expected_identity:
                raise EvidenceCorrupt("record_identity_path_mismatch")
            validate_canonical_admission(
                envelope.to_dict(), context=self._portable_context
            )
            record = envelope.to_record()
            return StoredTypedRecord(envelope, record, path)
        except EvidenceError:
            raise
        except PublicSafetyError as exc:
            raise EvidenceCorrupt(f"record_{exc.code}") from None
        except (
            CanonicalJsonError,
            ProjectionError,
            RecordValidationError,
            SafeIoError,
            OSError,
        ) as exc:
            raise EvidenceCorrupt("record_verification_failed") from exc

    def _read_bundle_manifest_path(
        self, path: Path, expected_identity: ContentIdentity
    ) -> BundleManifest:
        try:
            raw = read_stable_bytes(path)
            value = loads_canonical_json(raw)
            if not isinstance(value, Mapping):
                raise EvidenceCorrupt("bundle_manifest_not_object")
            manifest = BundleManifest.from_dict(value)
            if raw != dumps_canonical_json(manifest.to_dict()):
                raise EvidenceCorrupt("bundle_manifest_not_canonical")
            if manifest.identity != expected_identity:
                raise EvidenceCorrupt("bundle_manifest_path_mismatch")
            validate_canonical_admission(
                manifest.to_dict(), context=self._portable_context
            )
            return manifest
        except EvidenceError:
            raise
        except PublicSafetyError as exc:
            raise EvidenceCorrupt(f"bundle_manifest_{exc.code}") from None
        except (
            ArtifactError,
            CanonicalJsonError,
            SafeIoError,
            OSError,
        ) as exc:
            raise EvidenceCorrupt("bundle_manifest_verification_failed") from exc

    def _verify_reference_closure(self, records: tuple[StoredTypedRecord, ...]) -> None:
        index = {
            (stored.envelope.record_type, stored.envelope.identity.value): stored
            for stored in records
        }
        for stored in records:
            for dependency in _iter_typed_dependencies(
                stored.record, include_record=False
            ):
                if isinstance(dependency, RecordReference):
                    target = index.get(
                        (dependency.record_type, dependency.identity.value)
                    )
                    if target is None:
                        raise EvidenceCorrupt("dangling_record_reference")
                    if target.reference.logical_id != dependency.logical_id:
                        raise EvidenceCorrupt("record_reference_logical_id_mismatch")
                else:
                    target = index.get(
                        (dependency.RECORD_TYPE, dependency.identity.value)
                    )
                    if target is None:
                        raise EvidenceCorrupt("missing_embedded_record")

    def _project_path(self, value: Path | str) -> Path:
        root = Path(os.path.abspath(self.project_root))
        raw = Path(value)
        candidate = raw if raw.is_absolute() else root / raw
        candidate = Path(os.path.abspath(candidate))
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise EvidenceError("evidence_path_outside_project") from exc
        return candidate

    def _require_project_root(self) -> None:
        if not _safe_exists(self.project_root):
            raise EvidenceNotFound("project_not_found")
        try:
            require_safe_directory(self.project_root)
            if not _safe_exists(self.layout.root):
                raise EvidenceNotFound("store_not_found")
            require_safe_directory(self.layout.root)
        except EvidenceError:
            raise
        except (OSError, UnsafeFilesystemPath) as exc:
            raise EvidenceCorrupt("unsafe_project_root") from exc

    def _verify_store_surfaces(self) -> None:
        allowed_root = {"derived", "immutable", "streams"}
        allowed_immutable = {"bundle-manifests", "records"}
        try:
            root_entries = sorted(
                self.layout.root.iterdir(), key=lambda path: path.name
            )
            for entry in root_entries:
                require_safe_directory(entry)
            if not {entry.name for entry in root_entries} <= allowed_root:
                raise EvidenceCorrupt("unsupported_store_surface")
            immutable = self.layout.root / "immutable"
            if _safe_exists(immutable):
                require_safe_directory(immutable)
                immutable_entries = sorted(
                    immutable.iterdir(), key=lambda path: path.name
                )
                for entry in immutable_entries:
                    require_safe_directory(entry)
                if not {entry.name for entry in immutable_entries} <= allowed_immutable:
                    raise EvidenceCorrupt("unsupported_immutable_surface")
        except EvidenceError:
            raise
        except (OSError, UnsafeFilesystemPath) as exc:
            raise EvidenceCorrupt("unsafe_store_layout") from exc


def _safe_exists(path: Path) -> bool:
    try:
        return safe_path_stat(path, allow_missing=True) is not None
    except UnsafeFilesystemPath as exc:
        raise EvidenceCorrupt("unsafe_evidence_path") from exc


def _iter_typed_dependencies(
    value: Any, *, include_record: bool = True
) -> Iterator[RecordReference | TypedRecord]:
    """Walk declared dataclass fields while treating arbitrary JSON as opaque."""

    if isinstance(value, RecordReference):
        yield value
        return
    if isinstance(value, TypedRecord):
        if include_record:
            yield value
        if is_dataclass(value):
            for field in fields(value):
                yield from _iter_typed_dependencies(getattr(value, field.name))
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from _iter_typed_dependencies(getattr(value, field.name))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_typed_dependencies(item)


def _contains_untyped_evidence_shape(value: Any) -> bool:
    """Reject record wire shapes in events until typed event schemas own them."""

    if isinstance(value, Mapping):
        if set(value) in (_REFERENCE_WIRE_FIELDS, _ENVELOPE_WIRE_FIELDS):
            return True
        return any(_contains_untyped_evidence_shape(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_untyped_evidence_shape(item) for item in value)
    return False


def _empty_stream_state() -> dict[str, Any]:
    return {"event_count": 0, "event_types": {}, "head_identity": None}


def _reduce_stream_state(
    state: Mapping[str, Any], event: StoredEvent
) -> dict[str, Any]:
    event_types = dict(state["event_types"])
    event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    return {
        "event_count": state["event_count"] + 1,
        "event_types": dict(sorted(event_types.items())),
        "head_identity": {
            "algorithm": event.identity.algorithm,
            "value": event.identity.value,
        },
    }
