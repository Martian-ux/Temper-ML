from pathlib import Path

import pytest

from temper_ml.store.layout import StoreLayout, StorePathError
from temper_ml.domain.projections import ContentIdentity


def test_layout_separates_immutable_events_and_records_from_mutable_state(
    tmp_path: Path,
) -> None:
    layout = StoreLayout(tmp_path)

    assert (
        layout.run_events("run-1") == tmp_path / ".temper" / "runs" / "run-1" / "events"
    )
    assert (
        layout.run_state("run-1")
        == tmp_path / ".temper" / "runs" / "run-1" / "state.json"
    )
    assert layout.registry_events() == tmp_path / ".temper" / "registry" / "events"
    assert layout.registry_state() == tmp_path / ".temper" / "registry" / "state.json"
    assert layout.artifact_record("artifact-1") == (
        tmp_path / ".temper" / "artifacts" / "artifact-1" / "artifact.json"
    )
    identity = ContentIdentity("sha256", "a" * 64)
    assert layout.record_path("project", identity) == (
        tmp_path
        / ".temper"
        / "immutable"
        / "records"
        / "project"
        / "sha256"
        / f"{identity.value}.json"
    )
    assert layout.stream_events("project-lifecycle") == (
        tmp_path / ".temper" / "streams" / "project-lifecycle" / "events"
    )
    assert layout.stream_state("project-lifecycle") == (
        tmp_path
        / ".temper"
        / "derived"
        / "streams"
        / "project-lifecycle"
        / "state.json"
    )
    assert layout.bundle_manifest_path(identity) == (
        tmp_path
        / ".temper"
        / "immutable"
        / "bundle-manifests"
        / "sha256"
        / f"{identity.value}.json"
    )


@pytest.mark.parametrize(
    "component",
    [
        "",
        ".",
        "..",
        "/absolute",
        "\\absolute",
        "two/parts",
        "two\\parts",
        "C:drive",
        "nul",
        "name.",
        "name ",
        "control\x00",
    ],
)
def test_layout_rejects_unsafe_logical_ids(tmp_path: Path, component: str) -> None:
    layout = StoreLayout(tmp_path)
    with pytest.raises(StorePathError):
        layout.run_events(component)
