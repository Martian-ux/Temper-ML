from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from temper_ml.domain.records import CORE_PROJECTION_REGISTRY
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.redaction import (
    PublicSafetyError,
    RedactionContext,
    public_export_wrapper,
    redact_for_public_export,
    validate_canonical_admission,
)


CONTEXT = RedactionContext(
    local_usernames=("synthetic-user",),
    local_hostnames=("synthetic-host",),
)
REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ({"nested": {"api_key": "inert-secret"}}, "secret_field"),
        ({"nested": {"apiToken": "inert-secret"}}, "secret_field"),
        ({"nested": {"APIToken": "inert-secret"}}, "secret_field"),
        ({"url": "https://private.invalid/resource"}, "url"),
        ({"endpoint": "postgresql://fixture.invalid/database"}, "url"),
        ({"path_value": "C:\\synthetic\\private"}, "absolute_path"),
        ({"path_value": "\\\\synthetic-host\\share"}, "absolute_path"),
        ({"path_value": "~synthetic-user/private"}, "absolute_path"),
        ({"path_value": "//synthetic/share"}, "absolute_path"),
        ({"path_value": "\\synthetic\\rooted"}, "absolute_path"),
        ({"text": "config=C:\\synthetic\\private"}, "absolute_path"),
        ({"text": "config=/synthetic/private"}, "absolute_path"),
        ({"text": "stored under /synthetic/private"}, "absolute_path"),
        ({"path": "/etc/passwd"}, "absolute_path"),
        ({"path": "/recipe_resolution/private-client-key"}, "absolute_path"),
        ({"email": "person@example.invalid"}, "email"),
        ({"network": "192.0.2.10"}, "ip_address"),
        ({"network": "00:11:22:33:44:55"}, "mac_address"),
        ({"text": "owned by synthetic-user"}, "local_username"),
        ({"text": "worker synthetic-host"}, "local_hostname"),
        ({"visibility": "private", "reference": "opaque"}, "private_reference"),
        ({"text": "unsafe\u202esequence"}, "control_text"),
        ({"text": "sk-" + "x" * 24}, "secret_value"),
        ({"C:\\synthetic\\private": "value"}, "key_absolute_path"),
    ],
)
def test_canonical_admission_rejects_private_values_without_echoing_them(
    value, code: str
) -> None:
    with pytest.raises(PublicSafetyError) as caught:
        validate_canonical_admission(value, context=CONTEXT)

    assert caught.value.code == code
    assert str(caught.value) == code
    assert "synthetic-user" not in str(caught.value)


def test_canonical_admission_allows_json_pointer_and_explicit_public_url() -> None:
    validate_canonical_admission(
        {
            "path": "/recipe_resolution/identity/value",
            "documentation": "https://docs.example.invalid/public",
        },
        context=RedactionContext(
            allowed_public_url_prefixes=("https://docs.example.invalid/",)
        ),
    )


def test_url_allowlist_checks_every_uri_and_exact_origin() -> None:
    allowed = RedactionContext(
        allowed_public_url_prefixes=("https://docs.example.invalid",)
    )
    validate_canonical_admission(
        {"documentation": "https://docs.example.invalid/public"}, context=allowed
    )

    with pytest.raises(PublicSafetyError, match="^url$"):
        validate_canonical_admission(
            {
                "documentation": (
                    "https://docs.example.invalid/public and "
                    "https://private.invalid/resource"
                )
            },
            context=allowed,
        )
    with pytest.raises(PublicSafetyError, match="^url$"):
        validate_canonical_admission(
            {"documentation": "https://docs.example.invalid.attacker.invalid/value"},
            context=allowed,
        )


def test_default_admission_is_host_independent() -> None:
    validate_canonical_admission(
        {"text": "synthetic-user and synthetic-host are ordinary fixture text"}
    )


def test_public_projection_strips_identity_references_and_arbitrary_private_data() -> (
    None
):
    sensitive = "synthetic-sensitive-input"
    source = {
        "record_type": "local_use_session",
        "schema_version": "v1",
        "projection_version": "v1",
        "fields": {
            "session_id": "session-private",
            "display_name": "Private session name",
            "artifact": {
                "record_type": "artifact",
                "logical_id": "artifact-private",
                "identity": {"algorithm": "sha256", "value": "a" * 64},
            },
            "runtime_identity": {"algorithm": "sha256", "value": "b" * 64},
            "inference_settings": {"private_url": "https://private.invalid"},
            "inputs": [{"text": sensitive}],
            "outputs": [{"text": "synthetic output"}],
            "safe_enum": "fixture",
        },
    }
    original = deepcopy(source)

    result = redact_for_public_export(source, context=CONTEXT)
    encoded = dumps_canonical_json(result.value)

    assert source == original
    assert result.redaction_count == len(source["fields"])
    assert sensitive.encode() not in encoded
    assert hashlib.sha256(sensitive.encode()).hexdigest().encode() not in encoded
    assert b"artifact-private" not in encoded
    assert b'"fields":{}' in encoded
    assert b'"source_schema_version":"v1"' in encoded
    assert b'"public_record_projection_version":"v1"' in encoded
    assert b"safe_enum" not in encoded


def test_public_export_wrapper_is_versioned_and_contains_no_source_location() -> None:
    first = public_export_wrapper(
        (
            {
                "record_type": "project",
                "schema_version": "v1",
                "projection_version": "v1",
                "fields": {"project_id": "project-private"},
            },
        ),
        record_counts={"project": 1},
        stream_summaries=({"stream_ordinal": 1, "event_count": 2},),
        context=CONTEXT,
    )
    second = public_export_wrapper(
        (
            {
                "record_type": "project",
                "schema_version": "v1",
                "projection_version": "v1",
                "fields": {"project_id": "project-private"},
            },
        ),
        record_counts={"project": 1},
        stream_summaries=({"stream_ordinal": 1, "event_count": 2},),
        context=CONTEXT,
    )

    assert dumps_canonical_json(first.value) == dumps_canonical_json(second.value)
    assert first.value["classification"] == "public_projection"
    assert first.value["allowed_locations"] == [
        "public_documentation",
        "public_issue",
        "public_repository",
    ]
    assert first.value["records"][0]["source_schema_version"] == "v1"
    assert "schema_version" not in first.value["records"][0]
    assert "project_root" not in first.value
    assert "source_identity" not in first.value

    schema = json.loads(
        (REPO_ROOT / "schemas" / "public" / "public-dump-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(first.value) == set(schema["required"])
    assert schema["$defs"]["recordType"]["enum"] == [
        registration.record_type
        for registration in CORE_PROJECTION_REGISTRY.registrations
    ]


def test_dataset_public_projection_is_registered_but_fail_closed_on_all_fields() -> (
    None
):
    source = {
        "record_type": "dataset_version",
        "schema_version": "v1",
        "projection_version": "v1",
        "fields": {
            "version_id": "dataset-private",
            "source": {"source_identity": {"algorithm": "sha256", "value": "a" * 64}},
            "accepted_examples": [{"rendered_identity": "b" * 64}],
            "exclusions": [{"source_ordinal": 1, "reason_code": "invalid"}],
        },
    }

    result = redact_for_public_export(source, context=CONTEXT)

    assert result.value == {
        "record_type": "dataset_version",
        "source_schema_version": "v1",
        "source_projection_version": "v1",
        "public_record_projection_version": "v1",
        "fields": {},
    }
    assert result.redaction_count == len(source["fields"])


def test_runtime_request_public_projection_is_registered_and_value_free() -> None:
    source = {
        "record_type": "resolved_runtime_request",
        "schema_version": "v1",
        "projection_version": "v1",
        "fields": {
            "request_id": "request-private",
            "experiment_manifest_identity": {
                "algorithm": "sha256",
                "value": "a" * 64,
            },
            "evaluation_mode": "no_quality_evaluation",
        },
    }

    result = redact_for_public_export(source, context=CONTEXT)

    assert result.value["record_type"] == "resolved_runtime_request"
    assert result.value["fields"] == {}
    assert result.redaction_count == len(source["fields"])
