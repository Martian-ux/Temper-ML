"""Canonical admission checks and separate public-safe projections."""

from __future__ import annotations

from dataclasses import dataclass
import getpass
import ipaddress
import re
import socket
from typing import Any, Iterable, Mapping
import unicodedata
from urllib.parse import urlsplit

from temper_ml.store.canonical_json import dumps_canonical_json

REDACTION_POLICY_VERSION = "v1"
PUBLIC_PROJECTION_SCHEMA_VERSION = "v1"

_SECRET_SEGMENTS = {
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "password",
    "privatekey",
    "secret",
}
_OPERATIONAL_KEYS = {
    "account_id",
    "device_id",
    "hostname",
    "ip_address",
    "mac_address",
    "machine_id",
    "organization_id",
    "pid",
    "process_id",
    "serial",
    "serial_number",
    "username",
}
_PUBLIC_RECORD_FIELDS: dict[str, frozenset[str]] = {
    record_type: frozenset()
    for record_type in (
        "adapter_export",
        "artifact",
        "artifact_availability",
        "baseline_policy",
        "base_model_revision",
        "cleanup_receipt",
        "compatibility_group",
        "dataset_version",
        "execution_target",
        "evaluation_result",
        "evaluation_suite",
        "experiment",
        "experiment_derivation",
        "hardware_capability_profile",
        "hardware_requirements",
        "local_use_session",
        "manifest_diff",
        "project",
        "project_policy",
        "recipe",
        "recipe_resolution",
        "recommendation",
        "recommendation_policy",
        "resolved_runtime_request",
        "review",
        "run",
        "task_definition",
        "user_decision",
    )
}
_PUBLIC_AGGREGATE_KEYS = {"event_count"}
_PUBLIC_RECORD_INPUT_FIELDS = {
    "record_type",
    "schema_version",
    "projection_version",
    "fields",
}
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9:/])/(?!/)[^\s\"')]+")
_ROOTED_PATH = re.compile(r"(?<![A-Za-z0-9:/])(?:/{2}|\\+)")
_URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_EMAIL = re.compile(r"(?<![\w.-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_MAC = re.compile(
    r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])"
)
_IP_TOKEN = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f:.]{3,})(?![\w:])")
_BEARER = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_PEM = re.compile(r"-----BEGIN [A-Z0-9 ]*(?:PRIVATE KEY|CERTIFICATE)-----")
_SECRET_VALUE = re.compile(
    r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16})\b"
)
_HEX_DIGEST = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")
_BIDI_OR_TERMINAL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069]"
)
_MANIFEST_POINTER_ROOTS = {
    "base_model_revision",
    "compatibility_group",
    "dataset_version",
    "evaluation_policy",
    "execution_target",
    "hardware_requirements",
    "project",
    "project_policy",
    "recipe",
    "recipe_resolution",
    "task_definition",
    "tokenizer_identity",
}
_MANIFEST_POINTER_SEGMENTS = {
    "algorithm",
    "identity",
    "logical_id",
    "record_type",
    "value",
}


class PublicSafetyError(ValueError):
    """A public-safe error that exposes only a stable symbolic code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RedactionContext:
    """Host-local values and explicitly allowed public URL prefixes."""

    local_usernames: tuple[str, ...] = ()
    local_hostnames: tuple[str, ...] = ()
    allowed_public_url_prefixes: tuple[str, ...] = ()

    @classmethod
    def current(cls) -> "RedactionContext":
        usernames = _nonempty_unique((getpass.getuser(),))
        hostnames = _nonempty_unique((socket.gethostname(), socket.getfqdn()))
        return cls(usernames, hostnames, ())


@dataclass(frozen=True)
class RedactionResult:
    value: Any
    redaction_count: int


def validate_canonical_admission(
    value: Any, *, context: RedactionContext | None = None
) -> None:
    """Reject forbidden private/operational values without mutating evidence."""

    active = context if context is not None else RedactionContext()
    _validate_value(value, active, key=None)


def redact_for_public_export(
    value: Any, *, context: RedactionContext | None = None
) -> RedactionResult:
    """Create a separate public projection; never claim canonical identity."""

    active = context if context is not None else RedactionContext()
    if isinstance(value, Mapping) and set(value) == _PUBLIC_RECORD_INPUT_FIELDS:
        projected, count = _project_public_record(value)
    else:
        projected, count = _project_value(value, active, key=None)
    validate_canonical_admission(projected, context=active)
    dumps_canonical_json(projected)
    return RedactionResult(projected, count)


def public_export_wrapper(
    records: Iterable[Mapping[str, Any]],
    *,
    record_counts: Mapping[str, int],
    stream_summaries: Iterable[Mapping[str, Any]],
    context: RedactionContext | None = None,
) -> RedactionResult:
    """Build a deterministic public-only wrapper from already verified evidence."""

    active = context if context is not None else RedactionContext()
    if any(
        record_type not in _PUBLIC_RECORD_FIELDS
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for record_type, count in record_counts.items()
    ):
        raise PublicSafetyError("invalid_public_record_counts")
    projected_records: list[Any] = []
    redactions = 0
    for record in records:
        result = redact_for_public_export(record, context=active)
        projected_records.append(result.value)
        redactions += result.redaction_count
    projected_records.sort(key=dumps_canonical_json)
    projected_streams: list[Any] = []
    for stream in stream_summaries:
        summary = {
            key: value for key, value in stream.items() if key != "stream_ordinal"
        }
        if (
            set(summary) != {"event_count"}
            or isinstance(summary["event_count"], bool)
            or not isinstance(summary["event_count"], int)
            or summary["event_count"] < 0
        ):
            raise PublicSafetyError("invalid_public_stream_summary")
        result = redact_for_public_export(summary, context=active)
        projected_streams.append(result.value)
        redactions += result.redaction_count
    projected_streams.sort(key=dumps_canonical_json)
    projected_streams = [
        {**stream, "stream_ordinal": index}
        for index, stream in enumerate(projected_streams, 1)
    ]
    wrapper = {
        "schema_version": PUBLIC_PROJECTION_SCHEMA_VERSION,
        "redaction_policy_version": REDACTION_POLICY_VERSION,
        "classification": "public_projection",
        "allowed_locations": [
            "public_documentation",
            "public_issue",
            "public_repository",
        ],
        "record_counts": dict(sorted(record_counts.items())),
        "records": projected_records,
        "event_streams": projected_streams,
    }
    validate_canonical_admission(wrapper, context=active)
    return RedactionResult(wrapper, redactions)


def _validate_value(value: Any, context: RedactionContext, key: str | None) -> None:
    if isinstance(value, Mapping):
        if value.get("visibility") == "private" or value.get("private") is True:
            raise PublicSafetyError("private_reference")
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise PublicSafetyError("invalid_key")
            normalized = _normalize_key(raw_key)
            if _is_secret_key(normalized):
                raise PublicSafetyError("secret_field")
            if normalized in _OPERATIONAL_KEYS:
                raise PublicSafetyError("operational_identifier")
            key_code = _unsafe_string_code(raw_key, context, key=None)
            if key_code is not None:
                raise PublicSafetyError(f"key_{key_code}")
            _validate_value(item, context, raw_key)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_value(item, context, key)
        return
    if isinstance(value, str):
        code = _unsafe_string_code(value, context, key)
        if code is not None:
            raise PublicSafetyError(code)


def _project_value(
    value: Any, context: RedactionContext, key: str | None
) -> tuple[Any, int]:
    if isinstance(value, Mapping):
        if _looks_like_reference(value):
            record_type = value["record_type"]
            if record_type not in _PUBLIC_RECORD_FIELDS:
                return {"record_type": "<redacted:type>"}, 3
            return {"record_type": record_type}, 2
        if _looks_like_identity(value):
            return "<redacted:identity>", 1
        projected: dict[str, Any] = {}
        count = 0
        for raw_key, item in value.items():
            if (
                not isinstance(raw_key, str)
                or _normalize_key(raw_key) not in _PUBLIC_AGGREGATE_KEYS
            ):
                count += 1
                continue
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                count += 1
                continue
            projected[raw_key] = item
        return projected, count
    if isinstance(value, (list, tuple)):
        projected_items: list[Any] = []
        count = 0
        for item in value:
            child, child_count = _project_value(item, context, key)
            projected_items.append(child)
            count += child_count
        return projected_items, count
    if isinstance(value, str):
        code = _unsafe_string_code(value, context, key)
        if code is not None:
            return f"<redacted:{code}>", 1
        if _HEX_DIGEST.search(value):
            return "<redacted:identity>", 1
        return "<redacted:text>", 1
    return value, 0


def _project_public_record(value: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    record_type = value.get("record_type")
    source_schema = value.get("schema_version")
    source_projection = value.get("projection_version")
    source_fields = value.get("fields")
    if (
        not isinstance(record_type, str)
        or record_type not in _PUBLIC_RECORD_FIELDS
        or source_schema != "v1"
        or source_projection != "v1"
        or not isinstance(source_fields, Mapping)
    ):
        raise PublicSafetyError("unsupported_public_record")
    allowed = _PUBLIC_RECORD_FIELDS[record_type]
    projected_fields = {
        key: source_fields[key] for key in sorted(allowed) if key in source_fields
    }
    return (
        {
            "record_type": record_type,
            "source_schema_version": source_schema,
            "source_projection_version": source_projection,
            "public_record_projection_version": "v1",
            "fields": projected_fields,
        },
        len(source_fields) - len(projected_fields),
    )


def _unsafe_string_code(
    value: str, context: RedactionContext, key: str | None
) -> str | None:
    if _BIDI_OR_TERMINAL.search(value):
        return "control_text"
    if _PEM.search(value) or _BEARER.search(value) or _SECRET_VALUE.search(value):
        return "secret_value"
    if _EMAIL.search(value):
        return "email"
    if _MAC.search(value):
        return "mac_address"
    if (
        _WINDOWS_PATH.search(value)
        or _ROOTED_PATH.search(value)
        or re.match(r"^~[^/\\\s]*[/\\]", value)
    ):
        return "absolute_path"
    if _POSIX_PATH.search(value) and not (
        key == "path" and _is_manifest_json_pointer(value)
    ):
        return "absolute_path"
    for url_match in _URL.finditer(value):
        if not _allowed_public_url(
            url_match.group(0), context.allowed_public_url_prefixes
        ):
            return "url"
    for token in _IP_TOKEN.findall(value):
        try:
            ipaddress.ip_address(token.strip(".[]"))
        except ValueError:
            continue
        return "ip_address"
    folded = value.casefold()
    for username in context.local_usernames:
        if _contains_word(folded, username.casefold()):
            return "local_username"
    for hostname in context.local_hostnames:
        if hostname and hostname.casefold() in folded:
            return "local_hostname"
    return None


def _allowed_public_url(url: str, prefixes: tuple[str, ...]) -> bool:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    for prefix in prefixes:
        allowed = urlsplit(prefix)
        try:
            same_origin = (
                allowed.scheme == "https"
                and allowed.username is None
                and allowed.password is None
                and not allowed.query
                and not allowed.fragment
                and parsed.hostname == allowed.hostname
                and parsed.port == allowed.port
            )
        except ValueError:
            continue
        if not same_origin:
            continue
        allowed_path = allowed.path or "/"
        if allowed_path == "/" or parsed.path == allowed_path:
            return True
        if allowed_path.endswith("/") and parsed.path.startswith(allowed_path):
            return True
    return False


def _looks_like_reference(value: Mapping[str, Any]) -> bool:
    return set(value) == {"record_type", "logical_id", "identity"} and isinstance(
        value.get("record_type"), str
    )


def _looks_like_identity(value: Mapping[str, Any]) -> bool:
    return set(value) == {"algorithm", "value"} and isinstance(
        value.get("algorithm"), str
    )


def _normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", normalized)
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized)
    return re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")


def _is_secret_key(value: str) -> bool:
    compact = value.replace("_", "")
    segments = set(value.split("_"))
    credential_token = "token" in segments and bool(
        segments
        & {
            "access",
            "api",
            "auth",
            "authentication",
            "bearer",
            "credential",
            "id",
            "refresh",
            "secret",
            "service",
            "session",
        }
    )
    return (
        bool(segments & _SECRET_SEGMENTS)
        or credential_token
        or compact
        in {
            "apikey",
            "apitoken",
            "accesskey",
            "accesstoken",
            "authtoken",
            "bearertoken",
            "connectionstring",
            "credentialtoken",
            "idtoken",
            "privatekey",
            "refreshtoken",
            "sessiontoken",
            "signedurl",
        }
    )


def _contains_word(value: str, word: str) -> bool:
    if not word:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", value) is not None


def _is_manifest_json_pointer(value: str) -> bool:
    if not value.startswith("/") or any(character.isspace() for character in value):
        return False
    segments = [
        segment.replace("~1", "/").replace("~0", "~")
        for segment in value.split("/")[1:]
    ]
    return (
        bool(segments)
        and segments[0] in _MANIFEST_POINTER_ROOTS
        and all(segment in _MANIFEST_POINTER_SEGMENTS for segment in segments[1:])
    )


def _nonempty_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}))
