from http.client import HTTPConnection
import json
from pathlib import Path
import socket
import threading

from temper_ml.cli import main
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore
from temper_ml.ui.server import create_ui_server


def _post(server, path: str, body: dict[str, object]) -> dict[str, object]:
    port = server.server_port
    connection = HTTPConnection("127.0.0.1", port, timeout=90)
    connection.request(
        "POST",
        path,
        body=json.dumps(body),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": f"http://127.0.0.1:{port}",
            "X-Temper-CSRF": server.csrf_token,
        },
    )
    response = connection.getresponse()
    value = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 200, value
    assert value["ok"] is True
    return value["data"]


def _get_workspace(server) -> dict[str, object]:
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    connection.request("GET", "/api/v1/workspace")
    response = connection.getresponse()
    value = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 200
    return value["data"]


def test_slice_seven_fixture_journey_is_complete_through_ui_and_cli(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    original_create_connection = socket.create_connection

    def loopback_only(address, *args, **kwargs):
        host = address[0]
        if host not in {"127.0.0.1", "::1"}:
            raise AssertionError("Slice 7 attempted non-loopback network access")
        return original_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", loopback_only)
    server = create_ui_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _post(server, "/api/v1/setup", {})["result"]["status"] == "open"
        imported = _post(server, "/api/v1/dataset/import", {"format": "fixture"})
        assert imported["result"]["statistics"]["accepted_rows"] == 3
        resolved = _post(server, "/api/v1/candidates/resolve", {})
        assert len(resolved["result"]["candidates"]) == 2
        launched = _post(server, "/api/v1/runs/launch", {})
        assert all(item["verified_artifact"] for item in launched["result"]["runs"])

        compared = _post(
            server,
            "/api/v1/playground/compare",
            {
                "prompt": "Synthetic Slice 7 synchronized prompt",
                "maximum_tokens": 64,
                "seed": 17,
            },
        )
        assert len(compared["result"]["outputs"]) == 2
        reviewed = _post(
            server,
            "/api/v1/playground/reviews/solo",
            {
                "notes": "Both synthetic outputs satisfy the declared format.",
                "ratings": {"ember": 1, "slate": 1},
                "declaration": "I reviewed both synchronized outputs.",
            },
        )
        assert reviewed["result"]["structured_review"] is True
        evaluated = _post(server, "/api/v1/evaluation/run", {})
        assert evaluated["result"]["recommendation"]["confidence"] == "low"
        assert (
            "qualified_objective_tradeoff"
            in evaluated["result"]["recommendation"]["conflicts"]
        )

        selected = _post(
            server,
            "/api/v1/decisions",
            {"candidate_key": "ember", "status": "selected"},
        )
        assert selected["result"]["recommendation_unchanged"] is True
        focused = _post(
            server,
            "/api/v1/local-use/focused",
            {
                "candidate_key": "ember",
                "prompt": "Synthetic focused use",
                "maximum_tokens": 64,
                "seed": 17,
                "save": True,
            },
        )
        assert focused["result"]["general_chat"] is False
        batch = _post(
            server,
            "/api/v1/local-use/batch",
            {
                "candidate_key": "ember",
                "prompts": ["Synthetic batch one", "Synthetic batch two"],
                "maximum_tokens": 64,
                "seed": 17,
                "save": False,
            },
        )
        assert batch["result"]["batch_size"] == 2
        capture = _post(
            server,
            "/api/v1/evaluation/capture",
            {
                "review_identity": reviewed["result"]["review"]["identity"],
                "suite_kind": "development",
            },
        )
        assert capture["result"]["suite_state"] == "modified"
        exported = _post(server, "/api/v1/exports", {"candidate_key": "ember"})
        assert exported["result"]["status"] == "verified"
        assert exported["result"]["hosted_deployment"] is False

        workspace = _get_workspace(server)
        assert workspace["status"] == "verified"
        assert len(workspace["artifacts"]) == 2
        assert {run["status"] for run in workspace["runs"]} == {"completed"}
        assert all(
            any(event["type"] == "run_progress" for event in run["events"])
            for run in workspace["runs"]
        )
        assert all(
            any(event["type"] == "run_log" for event in run["events"])
            for run in workspace["runs"]
        )
        assert workspace["general_chat"] is False
        assert workspace["external_dashboard"] is False
        assert workspace["hosted_deployment"] is False
        assert workspace["local_use"]["deployment_ready"] is False

        assert main(["status", str(tmp_path)]) == 0
        cli_status = json.loads(capsys.readouterr().out)
        assert cli_status["status"] == "verified"
        assert cli_status["record_counts"]["artifact"] == 2
        assert (
            main(["project-status", str(tmp_path), "--id", "project-fixture-runtime"])
            == 0
        )
        cli_project = json.loads(capsys.readouterr().out)
        assert cli_project["status"] == "open"

        public = dumps_canonical_json(TypedEvidenceStore(tmp_path).public_dump().value)
        assert b"Synthetic Slice 7 synchronized prompt" not in public
        assert str(tmp_path).encode() not in public
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    restarted = create_ui_server(tmp_path, port=0)
    restarted_thread = threading.Thread(target=restarted.serve_forever, daemon=True)
    restarted_thread.start()
    try:
        workspace = _get_workspace(restarted)
        assert workspace["store"]["record_counts"]["artifact"] == 2
        assert workspace["dataset"]["prepared_bytes_available"] is False
        assert workspace["dataset"]["reimport_required"] is False
    finally:
        restarted.shutdown()
        restarted.server_close()
        restarted_thread.join(timeout=10)
