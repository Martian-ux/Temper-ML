"""Deterministic identities for artifact bytes and filesystem bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Iterable

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
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
