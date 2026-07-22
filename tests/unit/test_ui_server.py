from http.client import HTTPConnection
import json
from pathlib import Path
import threading
import time
from urllib.parse import quote

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.ui.server import create_ui_server
from temper_ml.app_services.reproduction import (
    ReplayExecutionRequest,
    ReproductionService,
)
from temper_ml.store.evidence import EvidenceError
from temper_ml.runtime.library_backend import LibraryRuntimeError


def _request(
    server,
    method: str,
    path: str,
    body: str | None = None,
    *,
    csrf: bool = True,
    origin: bool = True,
    content_type: str = "application/json",
):
    port = server.server_port
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = content_type
    if csrf:
        headers["X-Temper-CSRF"] = server.csrf_token
    if origin:
        headers["Origin"] = f"http://127.0.0.1:{port}"
    connection = HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    headers = dict(response.getheaders())
    connection.close()
    return response.status, headers, payload


@pytest.fixture
def ui_server(tmp_path: Path):
    server = create_ui_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_ui_server_binds_only_numeric_loopback(tmp_path: Path) -> None:
    with pytest.raises(ApplicationServiceError, match="ui_host_not_loopback"):
        create_ui_server(tmp_path, host="0.0.0.0", port=0)
    with pytest.raises(ApplicationServiceError, match="ui_host_not_loopback"):
        create_ui_server(tmp_path, host="localhost", port=0)


def test_ui_opens_before_project_directory_exists(tmp_path: Path) -> None:
    project = tmp_path / "new-project"
    server = create_ui_server(project, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, payload = _request(server, "GET", "/api/v1/workspace", origin=False)
        assert status == 200
        assert b'"status":"empty"' in payload
        assert not project.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_ui_shell_is_local_accessible_and_hardened(ui_server) -> None:
    status, headers, body = _request(ui_server, "GET", "/", origin=False)
    assert status == 200
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert b"Evaluation Playground" not in body
    assert b"Review outputs as evidence" in body
    assert b"Evidence-ready candidates" in body
    assert b'id="overview-candidates"' in body
    assert b'id="workflow-steps"' in body
    assert b'id="review-capture-selection"' in body
    assert b"Convert selected review to case" in body
    assert b"Inspect storage before cleanup" in body
    assert b'id="storage-entries"' in body
    assert b'id="cleanup-confirmation"' in body
    assert b'id="replay-plan-view"' in body
    assert b'id="mode-banner"' in body
    assert b"CHOOSE A MODE" in body
    assert b"Fixture demo" in body
    assert b"Real local LoRA training" in body
    assert b'id="dataset-file"' in body
    assert b"Glint-Research/Fable-5-traces" in body
    assert b'id="field-cot"' in body
    assert b'id="field-context" value="prompt"' in body
    assert b'id="field-completion" value="trace"' in body
    assert b'id="field-output" value="messages"' in body
    assert b"Fable-5 <code>pi_agent</code> viewer rows" in body
    assert b'id="dataset-row-limit" type="number" min="1" max="10000" value="4"' in body
    assert b'id="dataset-max-tokens" type="number" min="1" value="32768"' in body
    assert b'id="operation-rail"' in body
    assert b"general chat" not in body.lower()
    assert b'aria-live="polite"' in body


def test_ui_post_requires_origin_csrf_and_json(ui_server) -> None:
    status, _, _ = _request(ui_server, "POST", "/api/v1/setup", "{}", csrf=False)
    assert status == 403
    status, _, _ = _request(ui_server, "POST", "/api/v1/setup", "{}", origin=False)
    assert status == 403
    status, _, _ = _request(
        ui_server,
        "POST",
        "/api/v1/setup",
        "{}",
        content_type="text/plain",
    )
    assert status == 415


def test_rejected_ui_post_times_out_an_incomplete_body(ui_server) -> None:
    connection = HTTPConnection("127.0.0.1", ui_server.server_port, timeout=10)
    connection.putrequest("POST", "/api/v1/setup")
    connection.putheader("Origin", f"http://127.0.0.1:{ui_server.server_port}")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", "2")
    connection.endheaders()

    started = time.monotonic()
    response = connection.getresponse()
    payload = response.read()
    elapsed = time.monotonic() - started
    connection.close()

    assert response.status == 403
    assert b'"code":"csrf_token_invalid"' in payload
    assert elapsed < 2

    status, _, _ = _request(ui_server, "GET", "/api/v1/workspace", origin=False)
    assert status == 200


def test_ui_routes_call_services_and_get_remains_read_only(ui_server) -> None:
    status, _, before = _request(ui_server, "GET", "/api/v1/workspace", origin=False)
    assert status == 200
    assert b'"record_count":0' in before

    status, _, created = _request(ui_server, "POST", "/api/v1/setup", "{}")
    assert status == 200
    assert b'"ok":true' in created

    status, _, after = _request(ui_server, "GET", "/api/v1/workspace", origin=False)
    assert status == 200
    assert after != before


def test_file_import_bypasses_small_json_limit(ui_server, monkeypatch) -> None:
    observed: dict[str, object] = {}

    def import_dataset(**kwargs):
        observed.update(kwargs)
        return {"imported": True}

    monkeypatch.setattr(ui_server.journey, "import_dataset", import_dataset)
    body = b"x" * (1024 * 1024 + 1)
    options = quote(json.dumps({"context_field": "context", "row_limit": 16}))
    path = f"/api/v1/dataset/import-file?format=jsonl&options={options}"
    status, _, payload = _request(
        ui_server,
        "POST",
        path,
        body.decode("ascii"),
        content_type="application/octet-stream",
    )

    assert status == 200
    assert json.loads(payload)["data"]["result"] == {"imported": True}
    assert observed["source_format"] == "jsonl"
    assert len(observed["source_bytes"]) == len(body)
    assert observed["options"] == {"context_field": "context", "row_limit": 16}


def test_workspace_get_does_not_reconcile_a_pending_replay(
    ui_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journey = ui_server.journey
    journey.setup_project()
    journey.import_dataset()
    journey.resolve_candidates()
    journey.launch_candidates()
    prepared = journey.prepare_replay("ember", "strict_replay")
    draft = journey._replay_draft
    assert draft is not None
    service = ReproductionService(journey.project_root)
    original_append = service.store.append_event
    failed = False

    def lose_replay_terminal(stream_id: str, event_request: object) -> object:
        nonlocal failed
        if (
            getattr(event_request, "event_type", None) == "replay_execution_completed"
            and not failed
        ):
            failed = True
            raise EvidenceError("fixture_pending_replay")
        return original_append(stream_id, event_request)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "append_event", lose_replay_terminal)
    with pytest.raises(ApplicationServiceError, match="^fixture_pending_replay$"):
        service.execute(
            ReplayExecutionRequest(draft.plan, draft.launch, draft.candidate_key)
        )

    before = {
        path.relative_to(journey.project_root): path.read_bytes()
        for path in (journey.project_root / ".temper").rglob("*")
        if path.is_file()
    }
    status, _, payload = _request(ui_server, "GET", "/api/v1/workspace", origin=False)
    after = {
        path.relative_to(journey.project_root): path.read_bytes()
        for path in (journey.project_root / ".temper").rglob("*")
        if path.is_file()
    }

    assert status == 200
    assert after == before
    workspace = json.loads(payload)["data"]
    execution = next(
        item
        for item in workspace["reproduction"]["executions"]
        if item["run_id"] == prepared["run_id"]
    )
    assert execution["status"] == "running"


def test_retention_and_replay_routes_validate_and_delegate(
    ui_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, object]] = []
    monkeypatch.setattr(
        ui_server.journey,
        "preview_cleanup",
        lambda entry_ids: observed.append(("preview", entry_ids)) or {"planned": True},
    )
    monkeypatch.setattr(
        ui_server.journey,
        "execute_cleanup",
        lambda plan_id, *, confirm, entry_ids: (
            observed.append(("cleanup", (plan_id, entry_ids, confirm)))
            or {"outcome": "completed"}
        ),
    )
    monkeypatch.setattr(
        ui_server.journey,
        "prepare_replay",
        lambda candidate_key, mode: (
            observed.append(("plan_replay", (candidate_key, mode)))
            or {"status": "ready"}
        ),
    )
    monkeypatch.setattr(
        ui_server.journey,
        "execute_replay",
        lambda plan_id, *, run_id, candidate_key, mode: (
            observed.append(("execute_replay", (plan_id, run_id, candidate_key, mode)))
            or {"status": "completed"}
        ),
    )

    requests = (
        (
            "/api/v1/storage/cleanup/preview",
            {"entry_ids": ["entry-one", "entry-two"]},
        ),
        (
            "/api/v1/storage/cleanup/execute",
            {
                "plan_id": "cleanup-plan-one",
                "entry_ids": ["entry-one", "entry-two"],
                "confirm": True,
            },
        ),
        (
            "/api/v1/replays/plan",
            {"candidate_key": "ember", "mode": "strict_replay"},
        ),
        (
            "/api/v1/replays/execute",
            {
                "plan_id": "replay-one",
                "run_id": "run-replay-one",
                "candidate_key": "ember",
                "mode": "strict_replay",
            },
        ),
    )
    for path, body in requests:
        status, _, payload = _request(ui_server, "POST", path, json.dumps(body))
        assert status == 200
        assert json.loads(payload)["ok"] is True

    assert observed == [
        ("preview", ("entry-one", "entry-two")),
        ("cleanup", ("cleanup-plan-one", ("entry-one", "entry-two"), True)),
        ("plan_replay", ("ember", "strict_replay")),
        (
            "execute_replay",
            ("replay-one", "run-replay-one", "ember", "strict_replay"),
        ),
    ]


def test_storage_script_binds_controls_and_consent_to_exact_plans() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "temper_ml" / "ui" / "assets" / "app.js"
    ).read_text(encoding="utf-8")

    assert "cleanup_selection_plan_mismatch" in source
    assert "replay_controls_plan_mismatch" in source
    assert "renderedCleanupPlanKey" in source
    assert "button, input, select, textarea" in source
    assert "checkbox.disabled = Boolean(cleanupPlan)" in source
    assert "replayCandidate.disabled = Boolean(replayPlan)" in source
    assert "run_id: plan.run_id" in source


def test_real_ui_contract_uses_explicit_sources_and_hides_fixture_only_controls() -> (
    None
):
    root = Path(__file__).parents[2] / "src" / "temper_ml" / "ui" / "assets"
    script = (root / "app.js").read_text(encoding="utf-8")
    document = (root / "index.html").read_text(encoding="utf-8")

    assert "hugging_face_source_mode: huggingFaceMode" in script
    assert "analysis?.reason_counts?.length" in script
    assert "Array.isArray(result.candidates)" in script
    assert "const conflicts = recommendation.conflicts || []" in script
    assert "dataset.statistics.split_counts || []" in script
    assert '"Built-in fixture demo rows"' in script
    assert "new XMLHttpRequest()" in script
    assert "select.replaceChildren(...choices.map" in script
    assert "No verified real adapter yet" in script
    assert 'artifact.artifact_kind === "real_trained_lora_adapter"' in script
    assert 'fixture && value("dataset-source-kind") === "hugging_face"' in script
    assert 'data-panel="evaluate" data-fixture-only' in document
    assert 'data-panel="storage" data-fixture-only' in document
    assert 'id="hf-source-mode"' in document
    assert 'id="dataset-analysis"' in document


def test_library_runtime_errors_return_action_safe_conflicts(
    ui_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_preflight(_options) -> dict[str, object]:
        raise LibraryRuntimeError("library_transformers_unavailable")

    monkeypatch.setattr(ui_server.journey, "resolve_candidates", fail_preflight)
    status, _, payload = _request(
        ui_server,
        "POST",
        "/api/v1/candidates/resolve",
        json.dumps({"options": {}}),
    )

    assert status == 409
    assert json.loads(payload) == {
        "ok": False,
        "error": {"code": "library_transformers_unavailable"},
    }


def test_file_import_returns_application_service_errors(
    ui_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(**_values) -> dict[str, object]:
        raise ApplicationServiceError("dataset_store_unavailable")

    monkeypatch.setattr(ui_server.journey, "import_dataset", fail_import)
    status, _, payload = _request(
        ui_server,
        "POST",
        "/api/v1/dataset/import-file?format=json&options=%7B%7D",
        "[]",
        content_type="application/octet-stream",
    )

    assert status == 409
    assert json.loads(payload) == {
        "ok": False,
        "error": {"code": "dataset_store_unavailable"},
    }


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            ApplicationServiceError("workspace_projection_failed"),
            409,
            "workspace_projection_failed",
        ),
        (OSError("private store location"), 500, "filesystem_error"),
        (RuntimeError("private projection detail"), 500, "internal_error"),
    ],
)
def test_workspace_get_failures_use_stable_json_errors(
    ui_server,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    def fail_workspace() -> dict[str, object]:
        raise error

    monkeypatch.setattr(ui_server.journey, "workspace", fail_workspace)

    status, headers, payload = _request(
        ui_server, "GET", "/api/v1/workspace", origin=False
    )

    assert status == expected_status
    assert headers["Content-Type"] == "application/json"
    assert json.loads(payload) == {"ok": False, "error": {"code": expected_code}}
    assert b"private" not in payload


def test_ui_package_has_no_store_import() -> None:
    root = Path(__file__).parents[2] / "src" / "temper_ml" / "ui"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "temper_ml.store" not in source
