import json
from pathlib import Path
import socket

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.local_use import LocalUseService
from temper_ml.app_services.runs import RunLifecycleStatus, RunService
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


def test_fixture_workflow_rerun_reuses_exact_same_directory_evidence(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    root = tmp_path / "same-project"
    first, first_output = _run_workflow(root, capsys, monkeypatch)
    before_store = TypedEvidenceStore(root)
    before_records = tuple(
        (stored.envelope.record_type, stored.envelope.identity)
        for stored in before_store.iter_records()
    )
    before_streams = tuple(
        (
            stream.stream_id,
            tuple(event.identity for event in stream.events),
        )
        for stream in before_store.iter_streams()
    )

    second, second_output = _run_workflow(root, capsys, monkeypatch)
    after_store = TypedEvidenceStore(root)
    after_records = tuple(
        (stored.envelope.record_type, stored.envelope.identity)
        for stored in after_store.iter_records()
    )
    after_streams = tuple(
        (
            stream.stream_id,
            tuple(event.identity for event in stream.events),
        )
        for stream in after_store.iter_streams()
    )

    assert second == first
    assert second_output == first_output
    assert after_records == before_records
    assert after_streams == before_streams


def test_fixture_workflow_continues_after_an_exact_completed_run(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    root = tmp_path / "continued-project"
    original_focused = LocalUseService.focused

    def stop_after_run(self, request):
        del self, request
        raise ApplicationServiceError("synthetic_stop_after_run")

    monkeypatch.setattr(LocalUseService, "focused", stop_after_run)
    assert main(["fixture-workflow", str(root)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "status": "error",
        "code": "synthetic_stop_after_run",
    }
    assert captured.out == ""
    assert (
        RunService(root).status("run-fixture-runtime") is RunLifecycleStatus.COMPLETED
    )
    records = [stored.record for stored in TypedEvidenceStore(root).iter_records()]
    assert not any(isinstance(record, LocalUseSession) for record in records)
    assert not any(isinstance(record, AdapterExport) for record in records)

    monkeypatch.setattr(LocalUseService, "focused", original_focused)
    value, _ = _run_workflow(root, capsys, monkeypatch)

    assert value["status"] == "verified"
    records = [stored.record for stored in TypedEvidenceStore(root).iter_records()]
    assert sum(isinstance(record, Run) for record in records) == 1
    assert sum(isinstance(record, Artifact) for record in records) == 1
    assert sum(isinstance(record, LocalUseSession) for record in records) == 1
    assert sum(isinstance(record, AdapterExport) for record in records) == 1
