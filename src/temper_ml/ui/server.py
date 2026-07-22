"""Hardened loopback HTTP transport over Temper application services."""

from __future__ import annotations

from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
import json
from pathlib import Path
import secrets
import socket
import sys
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from temper_ml.app_services.consumer_journey import (
    ConsumerJourneyError,
    ConsumerJourneyService,
)
from temper_ml.app_services.datasets import DatasetAdapterError, DatasetPreflightError
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.records import RecordValidationError, parse_identity
from temper_ml.runtime.fixture_inference import FixtureInferenceError
from temper_ml.runtime.library_backend import LibraryRuntimeError
from temper_ml.runtime.preflight import PreflightError
from temper_ml.runtime.recipe_resolution import RecipeResolutionError


MAX_REQUEST_BYTES = 1024 * 1024
MAX_FILE_REQUEST_BYTES = 256 * 1024 * 1024
REJECT_BODY_DRAIN_TIMEOUT_SECONDS = 0.25
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
ASSET_TYPES = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


class TemperUiServer(HTTPServer):
    """Single-user loopback server with one service-owned journey session."""

    journey: ConsumerJourneyService
    csrf_token: str
    public_host: str


class TemperUiHandler(BaseHTTPRequestHandler):
    """Minimal JSON and asset routes with no canonical-store dependency."""

    server: TemperUiServer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if not self._valid_host():
            self._error(HTTPStatus.FORBIDDEN, "host_not_allowed")
            return
        path = urlsplit(self.path).path
        if path == "/":
            self._bytes(
                HTTPStatus.OK,
                _index_html(self.server.csrf_token),
                "text/html; charset=utf-8",
            )
            return
        if path.startswith("/assets/"):
            name = path.removeprefix("/assets/")
            content_type = ASSET_TYPES.get(name)
            if content_type is None:
                self._error(HTTPStatus.NOT_FOUND, "route_not_found")
                return
            try:
                data = files("temper_ml.ui.assets").joinpath(name).read_bytes()
            except (FileNotFoundError, OSError):
                self._error(HTTPStatus.NOT_FOUND, "asset_not_found")
                return
            self._bytes(HTTPStatus.OK, data, content_type, cache=True)
            return
        if path == "/api/v1/workspace":
            try:
                self._success(self.server.journey.workspace())
            except ApplicationServiceError as exc:
                self._error(HTTPStatus.CONFLICT, exc.code)
            except (OSError, UnicodeError):
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "filesystem_error")
            except Exception:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")
            return
        self._error(HTTPStatus.NOT_FOUND, "route_not_found")

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if not self._valid_host():
            self._reject_post(HTTPStatus.FORBIDDEN, "host_not_allowed")
            return
        if not self._valid_origin():
            self._reject_post(HTTPStatus.FORBIDDEN, "origin_not_allowed")
            return
        if not secrets.compare_digest(
            self.headers.get("X-Temper-CSRF", ""), self.server.csrf_token
        ):
            self._reject_post(HTTPStatus.FORBIDDEN, "csrf_token_invalid")
            return
        path = urlsplit(self.path).path
        if path == "/api/v1/dataset/import-file":
            self._import_file()
            return
        if self.headers.get_content_type() != "application/json":
            self._reject_post(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type_invalid")
            return
        length = self._content_length()
        if length is None:
            return
        try:
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "json_body_invalid")
            return
        if not isinstance(value, dict):
            self._error(HTTPStatus.BAD_REQUEST, "json_body_invalid")
            return
        try:
            result = self._dispatch(path, value)
            self._success(
                {"result": result, "workspace": self.server.journey.workspace()}
            )
        except DatasetPreflightError as exc:
            self._error(
                HTTPStatus.CONFLICT,
                exc.code,
                {
                    "analysis": exc.analysis.to_view(),
                    "action": (
                        "Adjust field mapping, length policy, or split weights and "
                        "retry."
                    ),
                },
            )
        except ConsumerJourneyError as exc:
            self._error(HTTPStatus.CONFLICT, exc.code, exc.details)
        except (
            ApplicationServiceError,
            DatasetAdapterError,
            FixtureInferenceError,
            LibraryRuntimeError,
            PreflightError,
            RecipeResolutionError,
        ) as exc:
            self._error(HTTPStatus.CONFLICT, exc.code)
        except (RecordValidationError, TypeError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, "request_invalid")
        except (OSError, UnicodeError):
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "filesystem_error")
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    def _dispatch(self, path: str, body: Mapping[str, Any]) -> dict[str, object]:
        journey = self.server.journey
        if path == "/api/v1/setup":
            _allow_fields(
                body,
                (
                    "mode",
                    "model_source",
                    "tokenizer_source",
                    "display_name",
                    "model_family",
                    "architecture",
                    "revision",
                    "license",
                    "target",
                ),
            )
            return journey.setup_project(
                mode=_optional_text(body, "mode", "fixture_demo"),
                model_source=_optional_nullable_text(body, "model_source"),
                tokenizer_source=_optional_nullable_text(body, "tokenizer_source"),
                display_name=_optional_text(body, "display_name", "Local model"),
                model_family=_optional_text(body, "model_family", "local-causal-lm"),
                architecture=_optional_text(body, "architecture", "causal-lm"),
                revision=_optional_text(body, "revision", "local-revision"),
                license_name=_optional_text(
                    body, "license", "user-confirmed-local-license"
                ),
                target=_optional_text(body, "target", "native_local"),
            )
        if path == "/api/v1/dataset/import":
            _allow_fields(body, ("format", "source", "options"))
            source_format = _optional_text(body, "format", "fixture")
            source = body.get("source")
            if source is not None and not isinstance(source, str):
                raise ValueError("source")
            options = body.get("options", {})
            if not isinstance(options, Mapping):
                raise ValueError("options")
            return journey.import_dataset(
                source_format=source_format,
                source_text=source,
                options=options,
            )
        if path == "/api/v1/candidates/resolve":
            _allow_fields(body, ("options",))
            options = body.get("options", {})
            if not isinstance(options, Mapping):
                raise ValueError("options")
            return journey.resolve_candidates(options)
        if path == "/api/v1/runs/launch":
            _require_fields(body, ())
            return journey.launch_candidates()
        if path == "/api/v1/runs/cancel":
            _require_fields(body, ())
            return journey.cancel_run()
        if path == "/api/v1/playground/compare":
            _allow_fields(body, ("prompt", "maximum_tokens", "seed"))
            return journey.compare(
                prompt=_required_text(body, "prompt"),
                maximum_tokens=_optional_int(body, "maximum_tokens", 64),
                seed=_optional_int(body, "seed", 17),
            )
        if path == "/api/v1/playground/reviews/solo":
            _require_fields(body, ("notes", "ratings", "declaration"))
            return journey.record_solo_review(
                notes=_required_text(body, "notes"),
                ratings=_ratings(body),
                declaration=_required_text(body, "declaration"),
            )
        if path == "/api/v1/playground/reviews/blind/prepare":
            _require_fields(body, ())
            return journey.prepare_blind_review()
        if path == "/api/v1/playground/reviews/blind/seal":
            _require_fields(body, ("notes", "ratings", "declaration"))
            return journey.seal_blind_review(
                notes=_required_text(body, "notes"),
                ratings=_ratings(body),
                declaration=_required_text(body, "declaration"),
            )
        if path == "/api/v1/playground/reviews/blind/reveal":
            _require_fields(body, ())
            return journey.reveal_blind_review()
        if path == "/api/v1/evaluation/run":
            _require_fields(body, ())
            return journey.evaluate_candidates()
        if path == "/api/v1/evaluation/capture":
            _allow_fields(body, ("review_identity", "suite_kind"))
            identity = body.get("review_identity")
            if not isinstance(identity, Mapping):
                raise ValueError("review_identity")
            return journey.capture_review(
                parse_identity(identity, field="review_identity"),
                suite_kind=_optional_text(body, "suite_kind", "development"),
            )
        if path == "/api/v1/decisions":
            _allow_fields(body, ("candidate_key", "status", "override_reason"))
            reason = body.get("override_reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("override_reason")
            return journey.record_decision(
                candidate_key=_required_text(body, "candidate_key"),
                status=_optional_text(body, "status", "selected"),
                override_reason=reason,
            )
        if path == "/api/v1/local-use/focused":
            _allow_fields(
                body,
                ("candidate_key", "prompt", "maximum_tokens", "seed", "save"),
            )
            return journey.focused_local_use(
                candidate_key=_required_text(body, "candidate_key"),
                prompt=_required_text(body, "prompt"),
                maximum_tokens=_optional_int(body, "maximum_tokens", 64),
                seed=_optional_int(body, "seed", 17),
                save=_optional_bool(body, "save", True),
            )
        if path == "/api/v1/local-use/batch":
            _allow_fields(
                body,
                ("candidate_key", "prompts", "maximum_tokens", "seed", "save"),
            )
            prompts = body.get("prompts")
            if not isinstance(prompts, list) or any(
                not isinstance(prompt, str) for prompt in prompts
            ):
                raise ValueError("prompts")
            return journey.batch_local_use(
                candidate_key=_required_text(body, "candidate_key"),
                prompts=tuple(prompts),
                maximum_tokens=_optional_int(body, "maximum_tokens", 64),
                seed=_optional_int(body, "seed", 17),
                save=_optional_bool(body, "save", False),
            )
        if path == "/api/v1/exports":
            _require_fields(body, ("candidate_key",))
            return journey.export_selected(
                candidate_key=_required_text(body, "candidate_key")
            )
        if path == "/api/v1/storage/cleanup/preview":
            _require_fields(body, ("entry_ids",))
            entry_ids = body.get("entry_ids")
            if not isinstance(entry_ids, list) or any(
                not isinstance(entry_id, str) or not entry_id for entry_id in entry_ids
            ):
                raise ValueError("entry_ids")
            return journey.preview_cleanup(tuple(entry_ids))
        if path == "/api/v1/storage/cleanup/execute":
            _require_fields(body, ("plan_id", "entry_ids", "confirm"))
            entry_ids = body.get("entry_ids")
            if not isinstance(entry_ids, list) or any(
                not isinstance(entry_id, str) or not entry_id for entry_id in entry_ids
            ):
                raise ValueError("entry_ids")
            confirm = body.get("confirm")
            if not isinstance(confirm, bool):
                raise ValueError("confirm")
            return journey.execute_cleanup(
                _required_text(body, "plan_id"),
                confirm=confirm,
                entry_ids=tuple(entry_ids),
            )
        if path == "/api/v1/replays/plan":
            _require_fields(body, ("candidate_key", "mode"))
            return journey.prepare_replay(
                _required_text(body, "candidate_key"),
                _required_text(body, "mode"),
            )
        if path == "/api/v1/replays/execute":
            _require_fields(body, ("plan_id", "run_id", "candidate_key", "mode"))
            return journey.execute_replay(
                _required_text(body, "plan_id"),
                run_id=_required_text(body, "run_id"),
                candidate_key=_required_text(body, "candidate_key"),
                mode=_required_text(body, "mode"),
            )
        raise ApplicationServiceError("route_not_found")

    def _import_file(self) -> None:
        if self.headers.get_content_type() != "application/octet-stream":
            self._reject_post(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type_invalid")
            return
        length = self._content_length(maximum=MAX_FILE_REQUEST_BYTES)
        if length is None:
            return
        try:
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            source_format = _query_text(query, "format")
            options_text = _query_text(query, "options", default="{}")
            options = json.loads(options_text)
            if not isinstance(options, dict):
                raise ValueError("options")
            data = self.rfile.read(length)
            if len(data) != length:
                raise OSError("incomplete request body")
            result = self.server.journey.import_dataset(
                source_format=source_format,
                source_bytes=data,
                options=options,
            )
            self._success(
                {"result": result, "workspace": self.server.journey.workspace()}
            )
        except DatasetPreflightError as exc:
            self._error(
                HTTPStatus.CONFLICT,
                exc.code,
                {
                    "analysis": exc.analysis.to_view(),
                    "action": (
                        "Adjust field mapping, length policy, or split weights and "
                        "retry."
                    ),
                },
            )
        except ConsumerJourneyError as exc:
            self._error(HTTPStatus.CONFLICT, exc.code, exc.details)
        except (
            ApplicationServiceError,
            DatasetAdapterError,
            LibraryRuntimeError,
        ) as exc:
            self._error(HTTPStatus.CONFLICT, exc.code)
        except (RecordValidationError, TypeError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, "request_invalid")
        except (OSError, UnicodeError):
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "filesystem_error")
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    def _content_length(self, *, maximum: int = MAX_REQUEST_BYTES) -> int | None:
        raw = self.headers.get("Content-Length")
        try:
            length = int(raw) if raw is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._error(HTTPStatus.LENGTH_REQUIRED, "content_length_required")
            return None
        if length > maximum:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large")
            return None
        return length

    def _reject_post(self, status: HTTPStatus, code: str) -> None:
        self._drain_bounded_request_body()
        self._error(status, code)

    def _drain_bounded_request_body(self) -> None:
        """Consume an already-sent small body so Windows closes cleanly."""

        raw = self.headers.get("Content-Length")
        try:
            length = int(raw) if raw is not None else -1
        except ValueError:
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            return
        original_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(REJECT_BODY_DRAIN_TIMEOUT_SECONDS)
            if len(self.rfile.read(length)) != length:
                self.close_connection = True
        except OSError:
            self.close_connection = True
        finally:
            try:
                self.connection.settimeout(original_timeout)
            except OSError:
                self.close_connection = True

    def _valid_host(self) -> bool:
        supplied = self.headers.get("Host", "")
        expected = _host_header(self.server.public_host, self.server.server_port)
        return secrets.compare_digest(supplied, expected)

    def _valid_origin(self) -> bool:
        host = self.headers.get("Host", "")
        expected = f"http://{host}"
        return secrets.compare_digest(self.headers.get("Origin", ""), expected)

    def _success(self, data: object) -> None:
        self._json(HTTPStatus.OK, {"ok": True, "data": data})

    def _error(
        self,
        status: HTTPStatus,
        code: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        error: dict[str, object] = {"code": code}
        if details:
            error["details"] = dict(details)
        self._json(status, {"ok": False, "error": error})

    def _json(self, status: HTTPStatus, value: object) -> None:
        payload = json.dumps(
            value,
            allow_nan=False,
            default=_json_transport_value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._bytes(status, payload, "application/json")

    def _bytes(
        self,
        status: HTTPStatus,
        value: bytes,
        content_type: str,
        *,
        cache: bool = False,
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Cache-Control", "public, max-age=300" if cache else "no-store"
        )
        self.end_headers()
        self.wfile.write(value)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _json_transport_value(value: object) -> str:
    if isinstance(value, Decimal) and value.is_finite():
        return str(value)
    raise TypeError(f"unsupported JSON transport value: {type(value).__name__}")


def create_ui_server(
    project_root: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> TemperUiServer:
    if host not in LOOPBACK_HOSTS:
        raise ApplicationServiceError("ui_host_not_loopback")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ApplicationServiceError("ui_port_invalid")
    server_type: type[TemperUiServer] = TemperUiServer
    if host == "::1":
        server_type = type(
            "TemperIpv6UiServer",
            (TemperUiServer,),
            {"address_family": socket.AF_INET6},
        )
    journey = ConsumerJourneyService(project_root)
    journey.reconcile_pending_operations()
    server = server_type((host, port), TemperUiHandler)
    server.journey = journey
    server.csrf_token = secrets.token_urlsafe(32)
    server.public_host = host
    return server


def serve_ui(
    project_root: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = create_ui_server(project_root, host=host, port=port)
    url = f"http://{_host_header(host, server.server_port)}"
    sys.stdout.write(f"Temper local lab: {url}\n")
    sys.stdout.flush()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def _index_html(csrf_token: str) -> bytes:
    template = (
        files("temper_ml.ui.assets").joinpath("index.html").read_text(encoding="utf-8")
    )
    return template.replace("{{CSRF_TOKEN}}", csrf_token).encode("utf-8")


def _host_header(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _require_fields(body: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    if set(body) != set(fields):
        raise ValueError("fields")


def _allow_fields(body: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    if set(body) - set(fields):
        raise ValueError("fields")


def _required_text(body: Mapping[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field)
    return value


def _optional_text(body: Mapping[str, Any], field: str, default: str) -> str:
    value = body.get(field, default)
    if not isinstance(value, str) or not value:
        raise ValueError(field)
    return value


def _optional_nullable_text(body: Mapping[str, Any], field: str) -> str | None:
    value = body.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field)
    return value


def _query_text(
    query: Mapping[str, list[str]], field: str, *, default: str | None = None
) -> str:
    values = query.get(field)
    if values is None:
        if default is None:
            raise ValueError(field)
        return default
    if len(values) != 1 or not values[0]:
        raise ValueError(field)
    return values[0]


def _optional_int(body: Mapping[str, Any], field: str, default: int) -> int:
    value = body.get(field, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(field)
    return value


def _optional_bool(body: Mapping[str, Any], field: str, default: bool) -> bool:
    value = body.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(field)
    return value


def _ratings(body: Mapping[str, Any]) -> dict[str, int]:
    value = body.get("ratings")
    if not isinstance(value, dict) or any(
        not isinstance(alias, str)
        or not isinstance(rating, int)
        or isinstance(rating, bool)
        for alias, rating in value.items()
    ):
        raise ValueError("ratings")
    return dict(value)
