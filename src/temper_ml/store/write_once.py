"""Write-once local evidence store primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    ensure_safe_directory,
    safe_path_stat,
)
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)
from temper_ml.store.safe_io import (
    SafeIoError,
    read_stable_bytes,
    replace_bytes,
    write_once_bytes,
)

_WINDOWS_RESERVED = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE
)


class WriteOnceError(RuntimeError):
    """Base error for write-once store operations."""


class WriteOnceExists(WriteOnceError):
    """Raised when immutable evidence already exists for an identity."""


class WriteOnceCorrupt(WriteOnceError):
    """Raised when existing immutable evidence no longer matches its identity."""


@dataclass(frozen=True)
class WrittenRecord:
    identity: ContentIdentity
    path: Path


class WriteOnceStore:
    """Small local store with immutable evidence and mutable derived state separated."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write_projected_json(
        self, area: str, projection: HashProjection, record: Mapping[str, Any]
    ) -> WrittenRecord:
        payload = dumps_canonical_json(record)
        canonical_record = loads_canonical_json(payload)
        if not isinstance(canonical_record, Mapping):
            raise WriteOnceError("immutable records must be JSON objects")
        identity = content_identity(projection, canonical_record)
        path = self._immutable_path(area, identity)
        try:
            existing = safe_path_stat(path, allow_missing=True)
        except UnsafeFilesystemPath as exc:
            raise WriteOnceError(str(exc)) from exc
        if existing is not None:
            self._verify_immutable_record(path, projection, identity)
            raise WriteOnceExists(f"immutable content already exists: {identity}")
        self._ensure_safe_directory(path.parent)
        try:
            write_once_bytes(path, payload)
        except FileExistsError:
            self._verify_immutable_record(path, projection, identity)
            raise WriteOnceExists(
                f"immutable content already exists: {identity}"
            ) from None
        except SafeIoError as exc:
            raise WriteOnceError("unable to write immutable content safely") from exc
        return WrittenRecord(identity=identity, path=path)

    def read_projected_json(
        self, area: str, projection: HashProjection, identity: ContentIdentity
    ) -> Mapping[str, Any]:
        return self._verify_immutable_record(
            self._immutable_path(area, identity), projection, identity
        )

    def _verify_immutable_record(
        self, path: Path, projection: HashProjection, identity: ContentIdentity
    ) -> Mapping[str, Any]:
        self._assert_no_symlinks(path)
        try:
            record = loads_canonical_json(read_stable_bytes(path))
        except FileNotFoundError:
            raise
        except (CanonicalJsonError, SafeIoError, OSError) as exc:
            raise WriteOnceCorrupt(
                f"immutable content is not canonical JSON: {identity}"
            ) from exc
        if not isinstance(record, Mapping):
            raise WriteOnceCorrupt(
                f"immutable content is not a JSON object: {identity}"
            )
        if content_identity(projection, record) != identity:
            raise WriteOnceCorrupt(
                f"identity path does not match stored content: {identity}"
            )
        return record

    def write_derived_state(self, name: str, state: Mapping[str, Any]) -> Path:
        path = self.root / "derived" / _safe_relative_path(name).with_suffix(".json")
        self._ensure_safe_directory(path.parent)
        try:
            safe_path_stat(path, allow_missing=True)
        except UnsafeFilesystemPath as exc:
            raise WriteOnceError(str(exc)) from exc
        try:
            replace_bytes(path, dumps_canonical_json(state))
        except SafeIoError as exc:
            raise WriteOnceError("unable to replace derived state safely") from exc
        return path

    def _immutable_path(self, area: str, identity: ContentIdentity) -> Path:
        return (
            self.root
            / "immutable"
            / _safe_relative_path(area)
            / identity.algorithm
            / (identity.value + ".json")
        )

    def _ensure_safe_directory(self, path: Path) -> None:
        try:
            relative_path = path.relative_to(self.root)
        except ValueError as exc:
            raise WriteOnceError(f"store path escapes root: {path}") from exc

        try:
            ensure_safe_directory(self.root / relative_path)
        except (OSError, UnsafeFilesystemPath) as exc:
            raise WriteOnceError(str(exc)) from exc

    def _assert_no_symlinks(self, path: Path) -> None:
        try:
            relative_path = path.relative_to(self.root)
        except ValueError as exc:
            raise WriteOnceError(f"store path escapes root: {path}") from exc

        try:
            safe_path_stat(self.root / relative_path)
        except UnsafeFilesystemPath as exc:
            raise WriteOnceError(str(exc)) from exc


def _safe_relative_path(value: str) -> Path:
    if value.startswith(("/", "\\")) or ":" in value:
        raise WriteOnceError(f"absolute paths are not allowed: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute():
        raise WriteOnceError(f"absolute paths are not allowed: {value!r}")
    parts = tuple(part for chunk in value.split("/") for part in chunk.split("\\"))
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise WriteOnceError(f"path traversal is not allowed: {value!r}")
    if any(
        any(ord(character) < 32 or ord(character) == 127 for character in part)
        or part.endswith((".", " "))
        or _WINDOWS_RESERVED.fullmatch(part)
        for part in parts
    ):
        raise WriteOnceError(f"non-portable store path: {value!r}")
    return Path(*parts)
