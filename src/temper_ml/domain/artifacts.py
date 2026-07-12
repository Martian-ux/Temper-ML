"""Deterministic identities for artifact bytes and filesystem bundles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, ClassVar, Iterable, Mapping

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    identity_fields,
    parse_identity,
    require_identifier,
    require_string_tuple,
)
from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    is_link_or_reparse,
    require_safe_directory,
    require_safe_regular_file,
    same_file_object,
)

BUNDLE_PROJECTION = HashProjection("artifact.bundle", "v1")
BUNDLE_SCHEMA_VERSION = "v1"


class ArtifactError(ValueError):
    """Raised when artifact bytes or bundle members are unsafe or malformed."""


class ArtifactContentKind(str, Enum):
    BYTES = "bytes"
    BUNDLE = "bundle"
    IMMUTABLE_UPSTREAM_REVISION = "immutable_upstream_revision"


class AvailabilityState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    REMOVED = "removed"
    EXTERNAL_ONLY = "external_only"


@dataclass(frozen=True)
class StorageReference:
    """Portable logical storage name, never a host path or retrieval URI."""

    provider: str
    logical_key: str

    def __post_init__(self) -> None:
        require_identifier("storage provider", self.provider)
        require_identifier("storage logical_key", self.logical_key)

    def to_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "logical_key": self.logical_key}


@dataclass(frozen=True)
class Artifact(TypedRecord):
    """Immutable adapter descriptor, integrity evidence, and lineage."""

    RECORD_TYPE: ClassVar[str] = "artifact"

    artifact_id: str
    project: RecordReference
    producing_run: RecordReference
    adapter_type: str
    content_kind: ArtifactContentKind
    content_identity: ContentIdentity
    base_model_revision: RecordReference
    tokenizer_identity: ContentIdentity
    compatibility_groups: tuple[RecordReference, ...]
    parent_artifacts: tuple[RecordReference, ...]
    storage_references: tuple[StorageReference, ...]
    integrity_evidence: ContentIdentity
    provenance: ContentIdentity
    lineage_evidence: ContentIdentity

    def __post_init__(self) -> None:
        require_identifier("artifact_id", self.artifact_id)
        for field, record_type in (
            ("project", "project"),
            ("producing_run", "run"),
            ("base_model_revision", "base_model_revision"),
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, RecordReference)
                or value.record_type != record_type
            ):
                raise RecordValidationError(f"{field} must reference {record_type}")
        require_identifier("adapter_type", self.adapter_type)
        if not isinstance(self.content_kind, ArtifactContentKind):
            raise RecordValidationError("content_kind is invalid")
        for field in (
            "content_identity",
            "tokenizer_identity",
            "integrity_evidence",
            "provenance",
            "lineage_evidence",
        ):
            if not isinstance(getattr(self, field), ContentIdentity):
                raise RecordValidationError(f"{field} must be a content identity")
        object.__setattr__(
            self,
            "compatibility_groups",
            _artifact_references(
                "compatibility_groups",
                self.compatibility_groups,
                "compatibility_group",
                non_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "parent_artifacts",
            _artifact_references(
                "parent_artifacts",
                self.parent_artifacts,
                "artifact",
                non_empty=False,
            ),
        )
        if (
            not isinstance(self.storage_references, tuple)
            or not self.storage_references
        ):
            raise RecordValidationError("storage_references must be a non-empty tuple")
        if any(
            not isinstance(reference, StorageReference)
            for reference in self.storage_references
        ):
            raise RecordValidationError("storage_references contains an invalid value")
        storage_keys = tuple(
            (reference.provider, reference.logical_key)
            for reference in self.storage_references
        )
        if len(set(storage_keys)) != len(storage_keys):
            raise RecordValidationError(
                "storage_references must not contain duplicates"
            )
        object.__setattr__(
            self,
            "storage_references",
            tuple(
                sorted(
                    self.storage_references,
                    key=lambda item: (item.provider, item.logical_key),
                )
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "project": self.project.to_dict(),
            "producing_run": self.producing_run.to_dict(),
            "adapter_type": self.adapter_type,
            "content_kind": self.content_kind.value,
            "content_identity": identity_fields(self.content_identity),
            "base_model_revision": self.base_model_revision.to_dict(),
            "tokenizer_identity": identity_fields(self.tokenizer_identity),
            "compatibility_groups": [
                reference.to_dict() for reference in self.compatibility_groups
            ],
            "parent_artifacts": [
                reference.to_dict() for reference in self.parent_artifacts
            ],
            "storage_references": [
                reference.to_dict() for reference in self.storage_references
            ],
            "integrity_evidence": identity_fields(self.integrity_evidence),
            "provenance": identity_fields(self.provenance),
            "lineage_evidence": identity_fields(self.lineage_evidence),
        }


@dataclass(frozen=True)
class ArtifactAvailability(TypedRecord):
    """One immutable availability observation, separate from the Artifact."""

    RECORD_TYPE: ClassVar[str] = "artifact_availability"

    availability_id: str
    artifact: RecordReference
    state: AvailabilityState
    available_byte_classes: tuple[str, ...]
    storage_references: tuple[StorageReference, ...]
    checkpoint_resumable: bool
    observed_content_identity: ContentIdentity
    supersedes: RecordReference | None = None

    def __post_init__(self) -> None:
        require_identifier("availability_id", self.availability_id)
        if (
            not isinstance(self.artifact, RecordReference)
            or self.artifact.record_type != "artifact"
        ):
            raise RecordValidationError("artifact must reference an artifact")
        if not isinstance(self.state, AvailabilityState):
            raise RecordValidationError("availability state is invalid")
        object.__setattr__(
            self,
            "available_byte_classes",
            require_string_tuple(
                "available_byte_classes",
                self.available_byte_classes,
                non_empty=False,
                sorted_values=True,
            ),
        )
        if not isinstance(self.storage_references, tuple) or any(
            not isinstance(reference, StorageReference)
            for reference in self.storage_references
        ):
            raise RecordValidationError("storage_references must be a tuple")
        storage_keys = tuple(
            (reference.provider, reference.logical_key)
            for reference in self.storage_references
        )
        if len(set(storage_keys)) != len(storage_keys):
            raise RecordValidationError(
                "storage_references must not contain duplicates"
            )
        object.__setattr__(
            self,
            "storage_references",
            tuple(
                sorted(
                    self.storage_references,
                    key=lambda item: (item.provider, item.logical_key),
                )
            ),
        )
        if not isinstance(self.checkpoint_resumable, bool):
            raise RecordValidationError("checkpoint_resumable must be a boolean")
        if not isinstance(self.observed_content_identity, ContentIdentity):
            raise RecordValidationError(
                "observed_content_identity must be a content identity"
            )
        if self.supersedes is not None and (
            not isinstance(self.supersedes, RecordReference)
            or self.supersedes.record_type != "artifact_availability"
        ):
            raise RecordValidationError(
                "supersedes must reference an artifact_availability"
            )
        if self.state in (AvailabilityState.UNAVAILABLE, AvailabilityState.REMOVED):
            if self.available_byte_classes or self.storage_references:
                raise RecordValidationError(
                    "unavailable artifacts cannot advertise retained locations or bytes"
                )
            if self.checkpoint_resumable:
                raise RecordValidationError(
                    "unavailable artifacts cannot be checkpoint-resumable"
                )
        if self.state is AvailabilityState.AVAILABLE and not self.storage_references:
            raise RecordValidationError(
                "available artifacts require a logical storage reference"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "availability_id": self.availability_id,
            "artifact": self.artifact.to_dict(),
            "state": self.state.value,
            "available_byte_classes": list(self.available_byte_classes),
            "storage_references": [
                reference.to_dict() for reference in self.storage_references
            ],
            "checkpoint_resumable": self.checkpoint_resumable,
            "observed_content_identity": identity_fields(
                self.observed_content_identity
            ),
            "supersedes": (
                self.supersedes.to_dict() if self.supersedes is not None else None
            ),
        }


def _artifact_references(
    field: str,
    values: tuple[RecordReference, ...],
    record_type: str,
    *,
    non_empty: bool,
) -> tuple[RecordReference, ...]:
    if not isinstance(values, tuple) or (non_empty and not values):
        raise RecordValidationError(f"{field} must be a tuple")
    if any(
        not isinstance(value, RecordReference) or value.record_type != record_type
        for value in values
    ):
        raise RecordValidationError(f"{field} must reference {record_type}")
    keys = tuple(value.identity for value in values)
    if len(set(keys)) != len(keys):
        raise RecordValidationError(f"{field} must not contain duplicates")
    return tuple(
        sorted(values, key=lambda item: (item.identity.value, item.logical_id))
    )


@dataclass(frozen=True)
class BundleMember:
    """One regular file represented in a canonical bundle manifest."""

    path: str
    identity: ContentIdentity
    size: int

    def projected_fields(self) -> dict[str, object]:
        return {
            "path": self.path,
            "identity": {
                "algorithm": self.identity.algorithm,
                "value": self.identity.value,
            },
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BundleMember":
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "identity",
            "size",
        }:
            raise ArtifactError("bundle member fields are invalid")
        path = value["path"]
        size = value["size"]
        identity = value["identity"]
        if not isinstance(path, str):
            raise ArtifactError("bundle member path is invalid")
        normalized = _validate_member_path(path)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArtifactError("bundle member size is invalid")
        if not isinstance(identity, Mapping):
            raise ArtifactError("bundle member identity is invalid")
        try:
            parsed_identity = parse_identity(identity, field="bundle member identity")
        except RecordValidationError as exc:
            raise ArtifactError("bundle member identity is invalid") from exc
        return cls(normalized, parsed_identity, size)


@dataclass(frozen=True)
class BundleManifest:
    """A canonical, versioned projection of a bundle's regular files."""

    schema_version: str
    projection_version: str
    members: tuple[BundleMember, ...]
    identity: ContentIdentity

    def projected_fields(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "projection_version": self.projection_version,
            "members": [member.projected_fields() for member in self.members],
        }

    def to_dict(self) -> dict[str, object]:
        value = self.projected_fields()
        value["identity"] = identity_fields(self.identity)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BundleManifest":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "projection_version",
            "members",
            "identity",
        }:
            raise ArtifactError("bundle manifest fields are invalid")
        if value["schema_version"] != BUNDLE_SCHEMA_VERSION:
            raise ArtifactError("unsupported bundle schema version")
        if value["projection_version"] != BUNDLE_PROJECTION.version:
            raise ArtifactError("unsupported bundle projection version")
        raw_members = value["members"]
        raw_identity = value["identity"]
        if not isinstance(raw_members, list) or not isinstance(raw_identity, Mapping):
            raise ArtifactError("bundle manifest content is invalid")
        members = tuple(BundleMember.from_dict(member) for member in raw_members)
        if tuple(sorted(members, key=lambda member: member.path)) != members:
            raise ArtifactError("bundle manifest members are not canonically ordered")
        paths = tuple(member.path for member in members)
        if len(set(paths)) != len(paths):
            raise ArtifactError("bundle manifest contains duplicate members")
        try:
            claimed = parse_identity(raw_identity, field="bundle manifest identity")
        except RecordValidationError as exc:
            raise ArtifactError("bundle manifest identity is invalid") from exc
        fields = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "projection_version": BUNDLE_PROJECTION.version,
            "members": [member.projected_fields() for member in members],
        }
        actual = content_identity(BUNDLE_PROJECTION, fields)
        if claimed != actual:
            raise ArtifactError("bundle manifest identity mismatch")
        return cls(
            schema_version=BUNDLE_SCHEMA_VERSION,
            projection_version=BUNDLE_PROJECTION.version,
            members=members,
            identity=claimed,
        )


@dataclass(frozen=True)
class _FileSnapshot:
    identity: ContentIdentity
    size: int


def byte_identity(data: bytes) -> ContentIdentity:
    """Return the SHA-256 identity of bytes without path-derived input."""

    return ContentIdentity("sha256", hashlib.sha256(data).hexdigest())


def file_identity(
    path: Path | str, *, chunk_size: int = 1024 * 1024
) -> ContentIdentity:
    """Hash one stable opened-file snapshot using bounded memory."""

    return _snapshot_file(Path(path), chunk_size=chunk_size).identity


def build_bundle_manifest(
    root: Path | str, member_paths: Iterable[str] | None = None
) -> BundleManifest:
    """Build a location-independent manifest for regular files below ``root``."""

    bundle_root = Path(root)
    try:
        require_safe_directory(bundle_root)
    except (OSError, UnsafeFilesystemPath) as exc:
        raise ArtifactError(
            f"bundle root must be a safe non-symlink directory: {exc}"
        ) from exc

    if member_paths is None:
        raw_paths = _enumerated_paths(bundle_root)
    else:
        raw_paths = list(member_paths)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path = _validate_member_path(raw_path)
        if path in seen:
            raise ArtifactError(f"duplicate normalized bundle member path: {path!r}")
        seen.add(path)
        normalized.append(path)

    members: list[BundleMember] = []
    for relative in normalized:
        candidate = bundle_root.joinpath(*PurePosixPath(relative).parts)
        try:
            snapshot = _snapshot_file(candidate)
        except FileNotFoundError as exc:
            raise ArtifactError(f"bundle member does not exist: {relative!r}") from exc
        members.append(BundleMember(relative, snapshot.identity, snapshot.size))

    ordered = tuple(sorted(members, key=lambda member: member.path))
    fields = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "projection_version": BUNDLE_PROJECTION.version,
        "members": [member.projected_fields() for member in ordered],
    }
    return BundleManifest(
        schema_version=BUNDLE_SCHEMA_VERSION,
        projection_version=BUNDLE_PROJECTION.version,
        members=ordered,
        identity=content_identity(BUNDLE_PROJECTION, fields),
    )


def _enumerated_paths(root: Path) -> list[str]:
    paths: list[str] = []
    pending: list[tuple[Path, str]] = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            require_safe_directory(directory)
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except (OSError, UnsafeFilesystemPath) as exc:
            raise ArtifactError(f"unsafe bundle directory: {exc}") from exc
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactError(
                    f"unable to inspect bundle member: {relative!r}"
                ) from exc
            if is_link_or_reparse(info):
                raise ArtifactError(
                    f"bundle member is a symlink or reparse point: {relative!r}"
                )
            if stat.S_ISDIR(info.st_mode):
                pending.append((Path(entry.path), relative))
            elif stat.S_ISREG(info.st_mode):
                paths.append(relative)
            else:
                raise ArtifactError(
                    f"bundle member is not a regular file: {relative!r}"
                )
    return paths


def _snapshot_file(path: Path, *, chunk_size: int = 1024 * 1024) -> _FileSnapshot:
    if chunk_size <= 0:
        raise ArtifactError("chunk_size must be positive")
    try:
        require_safe_regular_file(path)
    except UnsafeFilesystemPath as exc:
        raise ArtifactError(str(exc)) from exc

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactError(
            f"unable to open artifact member safely: {path.name!r}"
        ) from exc

    digest = hashlib.sha256()
    bytes_read = 0
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
            raise ArtifactError(
                f"artifact member is not a safe regular file: {path.name!r}"
            )
        while chunk := handle.read(chunk_size):
            bytes_read += len(chunk)
            digest.update(chunk)
        after = os.fstat(handle.fileno())

    if (
        _snapshot_signature(before) != _snapshot_signature(after)
        or bytes_read != after.st_size
    ):
        raise ArtifactError(f"artifact member changed during snapshot: {path.name!r}")
    try:
        current = require_safe_regular_file(path)
    except (OSError, UnsafeFilesystemPath) as exc:
        raise ArtifactError(
            f"artifact member changed during snapshot: {path.name!r}"
        ) from exc
    if not same_file_object(after, current) or after.st_size != current.st_size:
        raise ArtifactError(f"artifact member changed during snapshot: {path.name!r}")
    return _FileSnapshot(
        identity=ContentIdentity("sha256", digest.hexdigest()),
        size=after.st_size,
    )


def _snapshot_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _validate_member_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError("bundle member path must be a non-empty string")
    if "\\" in value:
        raise ArtifactError(
            f"backslashes are not allowed in bundle member paths: {value!r}"
        )
    if value.startswith("/") or os.path.isabs(value):
        raise ArtifactError(f"absolute bundle member paths are not allowed: {value!r}")
    if len(value) >= 2 and value[1] == ":":
        raise ArtifactError(
            f"drive-qualified bundle member paths are not allowed: {value!r}"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ArtifactError(f"unsafe bundle member path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ArtifactError(f"bundle member path is not normalized: {value!r}")
    return normalized
