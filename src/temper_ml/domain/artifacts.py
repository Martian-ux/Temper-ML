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


def byte_identity(data: bytes) -> ContentIdentity:
    """Return the SHA-256 identity of bytes without path-derived input."""

    return ContentIdentity("sha256", hashlib.sha256(data).hexdigest())


def file_identity(
    path: Path | str, *, chunk_size: int = 1024 * 1024
) -> ContentIdentity:
    """Stream a regular file into SHA-256 using bounded memory."""

    if chunk_size <= 0:
        raise ArtifactError("chunk_size must be positive")
    candidate = Path(path)
    mode = candidate.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ArtifactError(f"artifact member is a symlink: {candidate.name!r}")
    if not stat.S_ISREG(mode):
        raise ArtifactError(
            f"artifact member is not a regular file: {candidate.name!r}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return ContentIdentity("sha256", digest.hexdigest())


def build_bundle_manifest(
    root: Path | str, member_paths: Iterable[str] | None = None
) -> BundleManifest:
    """Build a location-independent manifest for regular files below ``root``."""

    bundle_root = Path(root)
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ArtifactError("bundle root must be a non-symlink directory")

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
            mode = candidate.lstat().st_mode
        except FileNotFoundError as exc:
            raise ArtifactError(f"bundle member does not exist: {relative!r}") from exc
        if stat.S_ISLNK(mode):
            raise ArtifactError(f"bundle member is a symlink: {relative!r}")
        if not stat.S_ISREG(mode):
            raise ArtifactError(f"bundle member is not a regular file: {relative!r}")
        members.append(
            BundleMember(relative, file_identity(candidate), candidate.stat().st_size)
        )

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
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ArtifactError(f"bundle member is a symlink: {relative!r}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ArtifactError(f"bundle member is not a regular file: {relative!r}")
        paths.append(relative)
    return paths


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
