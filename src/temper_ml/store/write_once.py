"""Write-once local evidence store primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from temper_ml.domain.projections import ContentIdentity, HashProjection, content_identity
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json


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
        identity = content_identity(projection, record)
        path = self._immutable_path(area, identity)
        payload = dumps_canonical_json(record)
        if path.exists():
            existing = loads_canonical_json(path.read_bytes())
            if content_identity(projection, existing) != identity:
                raise WriteOnceCorrupt(f"identity path does not match existing content: {identity}")
            raise WriteOnceExists(f"immutable content already exists: {identity}")
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        handle = os.open(path, flags, 0o644)
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return WrittenRecord(identity=identity, path=path)

    def read_projected_json(self, area: str, identity: ContentIdentity) -> Any:
        return loads_canonical_json(self._immutable_path(area, identity).read_bytes())

    def write_derived_state(self, name: str, state: Mapping[str, Any]) -> Path:
        path = self.root / "derived" / _safe_relative_path(name).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_bytes(dumps_canonical_json(state))
        os.replace(temp_path, path)
        return path

    def _immutable_path(self, area: str, identity: ContentIdentity) -> Path:
        return self.root / "immutable" / _safe_relative_path(area) / identity.algorithm / (
            identity.value + ".json"
        )


def _safe_relative_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise WriteOnceError(f"absolute paths are not allowed: {value!r}")
    parts = candidate.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise WriteOnceError(f"path traversal is not allowed: {value!r}")
    return candidate
