"""Validated paths for the bounded canonical local store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath


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


def _component(value: str) -> str:
    if not isinstance(value, str) or value in ("", ".", ".."):
        raise StorePathError(f"invalid store path component: {value!r}")
    if "/" in value or "\\" in value or ":" in value:
        raise StorePathError(f"invalid store path component: {value!r}")
    if PurePath(value).is_absolute():
        raise StorePathError(f"absolute store path component: {value!r}")
    return value
