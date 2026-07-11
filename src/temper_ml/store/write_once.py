"""Write-once local evidence store primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from temper_ml.domain.projections import ContentIdentity, HashProjection, content_identity
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
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
        if path.exists() or path.is_symlink():
            self._verify_immutable_record(path, projection, identity)
            raise WriteOnceExists(f"immutable content already exists: {identity}")
        self._ensure_safe_directory(path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            handle = os.open(path, flags, 0o644)
        except FileExistsError:
            self._verify_immutable_record(path, projection, identity)
            raise WriteOnceExists(f"immutable content already exists: {identity}") from None
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return WrittenRecord(identity=identity, path=path)

    def read_projected_json(
        self, area: str, projection: HashProjection, identity: ContentIdentity
    ) -> Mapping[str, Any]:
        return self._verify_immutable_record(self._immutable_path(area, identity), projection, identity)

    def _verify_immutable_record(
        self, path: Path, projection: HashProjection, identity: ContentIdentity
    ) -> Mapping[str, Any]:
        self._assert_no_symlinks(path)
        try:
            record = loads_canonical_json(path.read_bytes())
        except FileNotFoundError:
            raise
        except (CanonicalJsonError, OSError) as exc:
            raise WriteOnceCorrupt(f"immutable content is not canonical JSON: {identity}") from exc
        if not isinstance(record, Mapping):
            raise WriteOnceCorrupt(f"immutable content is not a JSON object: {identity}")
        if content_identity(projection, record) != identity:
            raise WriteOnceCorrupt(f"identity path does not match stored content: {identity}")
        return record

    def write_derived_state(self, name: str, state: Mapping[str, Any]) -> Path:
        path = self.root / "derived" / _safe_relative_path(name).with_suffix(".json")
        self._ensure_safe_directory(path.parent)
        if path.is_symlink():
            raise WriteOnceError(f"symbolic links are not allowed in the store: {path}")
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_bytes(dumps_canonical_json(state))
        os.replace(temp_path, path)
        return path

    def _immutable_path(self, area: str, identity: ContentIdentity) -> Path:
        return self.root / "immutable" / _safe_relative_path(area) / identity.algorithm / (
            identity.value + ".json"
        )

    def _ensure_safe_directory(self, path: Path) -> None:
        try:
            relative_path = path.relative_to(self.root)
        except ValueError as exc:
            raise WriteOnceError(f"store path escapes root: {path}") from exc

        current = self.root
        if current.is_symlink():
            raise WriteOnceError(f"symbolic links are not allowed in the store: {current}")
        if current.exists():
            if not current.is_dir():
                raise WriteOnceError(f"store root is not a directory: {current}")
        else:
            current.mkdir(parents=True, exist_ok=True)

        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise WriteOnceError(f"symbolic links are not allowed in the store: {current}")
            try:
                current.mkdir()
            except FileExistsError:
                if current.is_symlink() or not current.is_dir():
                    raise WriteOnceError(f"store path is not a directory: {current}") from None

    def _assert_no_symlinks(self, path: Path) -> None:
        try:
            relative_path = path.relative_to(self.root)
        except ValueError as exc:
            raise WriteOnceError(f"store path escapes root: {path}") from exc

        current = self.root
        if current.is_symlink():
            raise WriteOnceError(f"symbolic links are not allowed in the store: {current}")
        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise WriteOnceError(f"symbolic links are not allowed in the store: {current}")


def _safe_relative_path(value: str) -> Path:
    if value.startswith(("/", "\\")) or ":" in value:
        raise WriteOnceError(f"absolute paths are not allowed: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute():
        raise WriteOnceError(f"absolute paths are not allowed: {value!r}")
    parts = tuple(part for chunk in value.split("/") for part in chunk.split("\\"))
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise WriteOnceError(f"path traversal is not allowed: {value!r}")
    return Path(*parts)
