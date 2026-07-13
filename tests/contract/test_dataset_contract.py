import hashlib
import json
from pathlib import Path

import pytest

from temper_ml.app_services.datasets import DatasetImportRequest, DatasetService
from temper_ml.domain.datasets import (
    DeduplicationRule,
    FieldMapping,
    FilterRule,
    RendererSpec,
    SplitPart,
    SplitRule,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    CORE_LOGICAL_ID_FIELDS,
    CORE_PROJECTION_REGISTRY,
    RecordEnvelope,
    RecordValidationError,
)
from temper_ml.store.canonical_json import dumps_canonical_json


REPO_ROOT = Path(__file__).parents[2]


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


class CharacterTokenizer:
    identity = _identity("character-tokenizer-v1")

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text)


def _import(tmp_path: Path):
    request = DatasetImportRequest(
        version_id="dataset-contract",
        field_mapping=FieldMapping("prompt", "answer"),
        renderer=RendererSpec(),
        filter_rule=FilterRule(1, 1_000, 1_000),
        deduplication_rule=DeduplicationRule(),
        split_rule=SplitRule(
            23,
            (SplitPart("train", 9), SplitPart("validation", 1)),
        ),
        tokenizer=CharacterTokenizer(),
        preview_limit=1,
    )
    return DatasetService(tmp_path).import_json(
        b'[{"prompt":"Synthetic prompt","answer":"Synthetic answer"}]',
        request,
    )


def test_dataset_version_round_trips_through_registered_typed_envelope(
    tmp_path: Path,
) -> None:
    prepared = _import(tmp_path)
    envelope = prepared.version.to_envelope()

    assert envelope.record_type == "dataset_version"
    assert envelope.projection.label == "record.dataset_version@v1"
    assert CORE_LOGICAL_ID_FIELDS["dataset_version"] == "version_id"
    assert RecordEnvelope.from_dict(envelope.to_dict()) == envelope
    assert RecordEnvelope.from_dict(envelope.to_dict()).to_record() == prepared.version

    tampered = envelope.to_dict()
    tampered["payload"]["rendered_bytes_count"] += 1
    with pytest.raises(RecordValidationError):
        RecordEnvelope.from_dict(tampered)


def test_published_dataset_schema_and_projection_registry_match_code() -> None:
    manifest = json.loads(
        (
            REPO_ROOT / "schemas" / "identity-projections" / "core-domain-v1.json"
        ).read_text(encoding="utf-8")
    )
    registration = next(
        item
        for item in manifest["registrations"]
        if item["record_type"] == "dataset_version"
    )
    code = CORE_PROJECTION_REGISTRY.resolve("dataset_version", "v1")

    assert registration == {
        "record_type": "dataset_version",
        "record_schema_version": "v1",
        "projection_name": code.projection.name,
        "projection_version": code.projection.version,
        "record_schema": "schemas/records/dataset_version.schema.json",
    }
    schema = json.loads(
        (REPO_ROOT / registration["record_schema"]).read_text(encoding="utf-8")
    )
    payload = schema["allOf"][1]["properties"]["payload"]
    assert payload["additionalProperties"] is False
    assert set(payload["required"]) == set(payload["properties"])
    assert schema["$defs"]["splitRule"]["properties"]["algorithm"]["const"] == (
        "sha256_weighted_bucket@v1"
    )


def test_public_dump_supports_dataset_counts_but_exposes_no_dataset_fields(
    tmp_path: Path,
) -> None:
    prepared = _import(tmp_path)
    store = DatasetService(tmp_path).store

    public = store.public_dump()
    encoded = dumps_canonical_json(public.value)

    assert public.value["record_counts"] == {"dataset_version": 1}
    assert public.value["records"] == [
        {
            "record_type": "dataset_version",
            "source_schema_version": "v1",
            "source_projection_version": "v1",
            "public_record_projection_version": "v1",
            "fields": {},
        }
    ]
    assert b"Synthetic prompt" not in encoded
    assert b"Synthetic answer" not in encoded
    assert prepared.version.identity.value.encode() not in encoded
    public_schema = json.loads(
        (REPO_ROOT / "schemas" / "public" / "public-dump-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert public_schema["$defs"]["recordType"]["enum"] == [
        item.record_type for item in CORE_PROJECTION_REGISTRY.registrations
    ]
