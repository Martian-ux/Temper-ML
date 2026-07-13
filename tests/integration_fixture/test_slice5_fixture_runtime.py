import json
from pathlib import Path
import socket

from temper_ml.cli import main
from temper_ml.domain.artifacts import Artifact
from temper_ml.domain.local_use import AdapterExport, LocalUseSession
from temper_ml.domain.runs import ResolvedRuntimeRequest, Run
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore


def _run_workflow(root: Path, capsys, monkeypatch) -> tuple[dict[str, object], str]:
    def network_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("fixture workflow attempted network access")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    assert main(["fixture-workflow", str(root)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    value = json.loads(captured.out)
    assert captured.out == dumps_canonical_json(value).decode()
    return value, captured.out


def test_slice_five_cli_completes_the_offline_native_workflow(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    root = tmp_path / "fixture-project"
    value, output = _run_workflow(root, capsys, monkeypatch)

    assert value["status"] == "verified"
    assert value["evaluation_mode"] == "no_quality_evaluation"
    assert value["batch_output_count"] == 2
    assert value["public_projection_verified"] is True
    assert value["hosted_deployment"] is False
    assert value["deployment_ready"] is False
    assert str(root) not in output

    store = TypedEvidenceStore(root)
    verification = store.verify()
    records = [stored.record for stored in store.iter_records()]
    assert sum(isinstance(record, ResolvedRuntimeRequest) for record in records) == 1
    assert sum(isinstance(record, Run) for record in records) == 1
    assert sum(isinstance(record, Artifact) for record in records) == 1
    assert sum(isinstance(record, LocalUseSession) for record in records) == 1
    assert sum(isinstance(record, AdapterExport) for record in records) == 1
    assert verification.event_count > 0
    event_types = {
        event.event_type for stream in store.iter_streams() for event in stream.events
    }
    assert {
        "run_preflight_succeeded",
        "runtime_request_frozen",
        "run_launched",
        "run_progress",
        "run_checkpoint",
        "run_log",
        "artifact_ingestion_started",
        "artifact_ingestion_verified",
        "run_completed",
        "local_use_session_saved",
        "adapter_export_verified",
    } <= event_types
    public = dumps_canonical_json(store.public_dump().value)
    assert b"Synthetic focused prompt" not in public
    assert str(root).encode() not in public


def test_fixture_workflow_is_location_independent_and_repeatable(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    first, first_output = _run_workflow(tmp_path / "first", capsys, monkeypatch)
    second, second_output = _run_workflow(tmp_path / "second", capsys, monkeypatch)

    assert first == second
    assert first_output == second_output
    first_artifact = next(
        stored.record
        for stored in TypedEvidenceStore(tmp_path / "first").iter_records()
        if isinstance(stored.record, Artifact)
    )
    second_artifact = next(
        stored.record
        for stored in TypedEvidenceStore(tmp_path / "second").iter_records()
        if isinstance(stored.record, Artifact)
    )
    assert first_artifact.identity == second_artifact.identity
    assert first_artifact.content_identity == second_artifact.content_identity
