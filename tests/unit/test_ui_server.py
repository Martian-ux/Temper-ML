from http.client import HTTPConnection
from pathlib import Path
import threading
import time

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.ui.server import create_ui_server


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
    assert b'id="review-capture-selection"' in body
    assert b"Convert selected review to case" in body
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


def test_ui_package_has_no_store_import() -> None:
    root = Path(__file__).parents[2] / "src" / "temper_ml" / "ui"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "temper_ml.store" not in source
