"""Validated paths for the bounded canonical local store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
import re

from temper_ml.domain.projections import ContentIdentity

_WINDOWS_RESERVED = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE
)


class StorePathError(ValueError):
    """Raised when a logical identifier could escape the store layout."""


@dataclass(frozen=True)
class StoreLayout:
    """Path-focused helper for events, mutable state, and artifact records."""

    project_root: Path

    def __init__(self, project_root: Path | str) -> None:
        object.__setattr__(self, "project_root", Path(project_root))

    @property
    def root(self) -> Path:
        return self.project_root / ".temper"

    def run_events(self, run_id: str) -> Path:
        return self.root / "runs" / _component(run_id) / "events"

    def run_state(self, run_id: str) -> Path:
        return self.root / "runs" / _component(run_id) / "state.json"

    def registry_events(self) -> Path:
        return self.root / "registry" / "events"

    def registry_state(self) -> Path:
        return self.root / "registry" / "state.json"

    def artifact_record(self, artifact_id: str) -> Path:
        return self.root / "artifacts" / _component(artifact_id) / "artifact.json"

    def records_root(self) -> Path:
        return self.root / "immutable" / "records"

    def record_directory(self, record_type: str) -> Path:
        return self.records_root() / _component(record_type) / "sha256"

    def record_path(self, record_type: str, identity: ContentIdentity) -> Path:
        if not isinstance(identity, ContentIdentity):
            raise StorePathError("record identity must be a ContentIdentity")
        return self.record_directory(record_type) / f"{identity.value}.json"

    def streams_root(self) -> Path:
        return self.root / "streams"

    def stream_events(self, stream_id: str) -> Path:
        return self.streams_root() / _component(stream_id) / "events"

    def stream_state(self, stream_id: str) -> Path:
        return self.root / "derived" / "streams" / _component(stream_id) / "state.json"

    def bundle_manifests_root(self) -> Path:
        return self.root / "immutable" / "bundle-manifests" / "sha256"

    def bundle_manifest_path(self, identity: ContentIdentity) -> Path:
        if not isinstance(identity, ContentIdentity):
            raise StorePathError("bundle identity must be a ContentIdentity")
        return self.bundle_manifests_root() / f"{identity.value}.json"


def _component(value: str) -> str:
    if not isinstance(value, str) or value in ("", ".", ".."):
        raise StorePathError(f"invalid store path component: {value!r}")
    if "/" in value or "\\" in value or ":" in value:
        raise StorePathError(f"invalid store path component: {value!r}")
    if PurePath(value).is_absolute():
        raise StorePathError(f"absolute store path component: {value!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise StorePathError(f"control characters are not allowed: {value!r}")
    if value.endswith((".", " ")) or _WINDOWS_RESERVED.fullmatch(value):
        raise StorePathError(f"non-portable store path component: {value!r}")
    return value
