import hashlib
import json
from pathlib import Path

import pytest

from temper_ml.app_services.datasets import (
    CsvDatasetAdapter,
    DatasetAdapterError,
    DatasetImportRequest,
    DatasetService,
    HuggingFaceRowsDatasetAdapter,
    ImportedSource,
    JsonDatasetAdapter,
    JsonlDatasetAdapter,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.datasets import (
    DatasetAdapter,
    DeduplicationRule,
    ExclusionPhase,
    FieldMapping,
    FilterRule,
    RendererSpec,
    SourceDescriptor,
    SplitPart,
    SplitRule,
    renderer_identity,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


class WordTokenizer:
    def __init__(self, label: str = "tokenizer-v1") -> None:
        self._identity = _identity(label)

    @property
    def identity(self) -> ContentIdentity:
        return self._identity

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _request(
    version_id: str,
    *,
    mapping: FieldMapping | None = None,
    tokenizer: object | None = None,
    split_rule: SplitRule | None = None,
    maximum_characters: int | None = 200,
    maximum_tokens: int | None = None,
) -> DatasetImportRequest:
    return DatasetImportRequest(
        version_id=version_id,
        field_mapping=mapping or FieldMapping("instruction", "response", "context"),
        renderer=RendererSpec(),
        filter_rule=FilterRule(1, maximum_characters, maximum_tokens),
        deduplication_rule=DeduplicationRule(),
        split_rule=split_rule
        or SplitRule(17, (SplitPart("train", 8), SplitPart("validation", 2))),
        tokenizer=tokenizer or WordTokenizer(),  # type: ignore[arg-type]
        preview_limit=2,
    )


def test_pipeline_is_byte_identical_and_records_safe_row_dispositions(
    tmp_path: Path,
) -> None:
    rejected = "rejected-private-source-value"
    rows = [
        {"instruction": "Rewrite alpha", "context": "Fixture A", "response": "Alpha"},
        {"instruction": "Rewrite beta", "context": "Fixture B", "response": "Beta"},
        {"instruction": "Rewrite alpha", "context": "Fixture A", "response": "Alpha"},
        {"instruction": rejected, "context": "Fixture C"},
        {
            "instruction": "Long fixture",
            "context": "Fixture D",
            "response": "x" * 300,
        },
    ]
    source = json.dumps(rows, separators=(",", ":")).encode()
    service = DatasetService(tmp_path)

    first = service.import_json(source, _request("dataset-one"))
    second = service.import_json(source, _request("dataset-one"))

    assert first.version.identity == second.version.identity
    assert first.version.to_dict() == second.version.to_dict()
    assert first.rendered_bytes == second.rendered_bytes
    assert first.version.split_membership == second.version.split_membership
    assert first.version.statistics.source_rows == 5
    assert first.version.statistics.accepted_rows == 2
    assert first.version.statistics.excluded_rows == 3
    assert first.version.statistics.duplicate_rows == 1
    assert [receipt.reason_code for receipt in first.version.exclusions] == [
        "duplicate_rendered_text",
        "missing_mapped_field",
        "above_maximum_characters",
    ]
    assert first.version.exclusions[0].retained_source_ordinal == 1
    assert first.version.exclusions[1].phase is ExclusionPhase.VALIDATION
    assert first.version.exclusions[2].phase is ExclusionPhase.FILTERING
    assert len(first.previews) == 2

    envelope_bytes = dumps_canonical_json(first.version.to_dict())
    assert rejected.encode() not in envelope_bytes
    assert hashlib.sha256(rejected.encode()).hexdigest().encode() not in envelope_bytes
    lines = first.rendered_bytes.splitlines(keepends=True)
    assert len(lines) == 2
    rendered_texts = []
    for line in lines:
        decoded = loads_canonical_json(line)
        assert line == dumps_canonical_json(decoded)
        assert set(decoded) == {
            "rendered_identity",
            "source_ordinal",
            "split",
            "text",
        }
        rendered_texts.append(decoded["text"])
    assert [preview.text for preview in first.previews] == rendered_texts
    assert (
        len(
            [
                stored
                for stored in service.store.iter_records()
                if stored.envelope.record_type == "dataset_version"
            ]
        )
        == 1
    )


def test_all_explicit_local_adapters_produce_supported_versions(tmp_path: Path) -> None:
    service = DatasetService(tmp_path)
    json_result = service.import_json(
        b'[{"instruction":"One","response":"First"}]',
        _request("dataset-json", mapping=FieldMapping("instruction", "response")),
    )
    jsonl_result = service.import_jsonl(
        b'{"instruction":"One","response":"First"}\n',
        _request("dataset-jsonl", mapping=FieldMapping("instruction", "response")),
    )
    csv_result = service.import_csv(
        b"instruction,response\r\nOne,First\r\n",
        _request("dataset-csv", mapping=FieldMapping("instruction", "response")),
    )
    rows = [{"prompt": "One", "completion": "First", "metadata": {"fold": 1}}]
    hf_result = service.import_hugging_face_rows(
        rows,
        _request(
            "dataset-hf",
            mapping=FieldMapping("prompt", "completion"),
        ),
    )

    assert [
        result.version.source.adapter.value
        for result in (json_result, jsonl_result, csv_result, hf_result)
    ] == ["json@v1", "jsonl@v1", "csv@v1", "hugging_face_rows@v1"]
    assert (
        len(
            {
                result.version.identity
                for result in (json_result, jsonl_result, csv_result, hf_result)
            }
        )
        == 4
    )
    rows[0]["prompt"] = "Mutated after import"
    assert b"Mutated after import" not in hf_result.rendered_bytes


def test_import_source_rejects_descriptor_and_row_provenance_mismatches(
    tmp_path: Path,
) -> None:
    source_bytes = b'[{"instruction":"Original","response":"Stable"}]'
    source_identity = ContentIdentity(
        "sha256",
        hashlib.sha256(source_bytes).hexdigest(),
    )
    descriptor = SourceDescriptor(DatasetAdapter.JSON, source_identity, 1)
    rows = ({"instruction": "Original", "response": "Stable"},)
    forged_descriptor = ImportedSource(
        SourceDescriptor(DatasetAdapter.JSON, _identity("forged-source"), 1),
        rows,
        source_bytes,
    )
    mismatched_rows = ImportedSource(
        descriptor,
        ({"instruction": "Substituted", "response": "Content"},),
        source_bytes,
    )
    service = DatasetService(tmp_path)

    for version_id, source in (
        ("dataset-forged-descriptor", forged_descriptor),
        ("dataset-mismatched-rows", mismatched_rows),
    ):
        with pytest.raises(ApplicationServiceError, match="^dataset_source_invalid$"):
            service.import_source(
                source,
                _request(
                    version_id,
                    mapping=FieldMapping("instruction", "response"),
                ),
            )

    assert tuple(service.store.iter_records()) == ()


def test_imported_source_copies_mutable_rows_before_import(tmp_path: Path) -> None:
    source_bytes = (
        b'[{"instruction":"Original","metadata":{"tags":["one"]},"response":"Stable"}]'
    )
    mutable_row = {
        "instruction": "Original",
        "metadata": {"tags": ["one"]},
        "response": "Stable",
    }
    source = ImportedSource(
        SourceDescriptor(
            DatasetAdapter.JSON,
            ContentIdentity("sha256", hashlib.sha256(source_bytes).hexdigest()),
            1,
        ),
        (mutable_row,),
        source_bytes,
    )
    mutable_row["instruction"] = "Mutated after construction"
    metadata = mutable_row["metadata"]
    assert isinstance(metadata, dict)
    tags = metadata["tags"]
    assert isinstance(tags, list)
    tags.append("mutated")

    result = DatasetService(tmp_path).import_source(
        source,
        _request(
            "dataset-copied-source",
            mapping=FieldMapping("instruction", "response"),
        ),
    )

    assert b"Original" in result.rendered_bytes
    assert b"Mutated after construction" not in result.rendered_bytes
    assert source.rows[0]["metadata"] == {"tags": ("one",)}


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"])
def test_jsonl_treats_unicode_separators_as_json_string_content(
    tmp_path: Path,
    separator: str,
) -> None:
    source_bytes = (
        '{"instruction":"First' + separator + 'Second","response":"Stable"}\r\n'
    ).encode()

    result = DatasetService(tmp_path).import_jsonl(
        source_bytes,
        _request(
            "dataset-jsonl-unicode-" + format(ord(separator), "x"),
            mapping=FieldMapping("instruction", "response"),
        ),
    )

    assert separator.encode() in result.rendered_bytes
    assert result.version.source.row_count == 1


@pytest.mark.parametrize(
    ("adapter", "source", "code"),
    [
        (JsonDatasetAdapter(), b"{}", "json_rows_required"),
        (JsonDatasetAdapter(), b'[{"a":1,"a":2}]', "json_duplicate_key"),
        (JsonlDatasetAdapter(), b'{"a":1}\n\n{"a":2}\n', "jsonl_blank_line"),
        (CsvDatasetAdapter(), b"a,a\r\n1,2\r\n", "csv_header_invalid"),
        (CsvDatasetAdapter(), b"a,b\r\n1\r\n", "csv_row_width_invalid"),
        (HuggingFaceRowsDatasetAdapter(), ["not-a-row"], "hugging_face_rows_invalid"),
    ],
)
def test_adapters_fail_closed_with_stable_non_echoing_codes(
    adapter, source, code: str
) -> None:
    with pytest.raises(DatasetAdapterError) as caught:
        adapter.load(source)

    assert caught.value.code == code
    assert str(caught.value) == code
    assert "not-a-row" not in str(caught.value)


def test_complete_governed_inputs_change_their_bound_identities(tmp_path: Path) -> None:
    service = DatasetService(tmp_path)
    source = b'[{"instruction":"One","response":"First","alt":"Alternate"}]'
    base = service.import_json(
        source,
        _request("dataset-base", mapping=FieldMapping("instruction", "response")),
    )
    remapped = service.import_json(
        source,
        _request("dataset-remapped", mapping=FieldMapping("alt", "response")),
    )
    retokenized = service.import_json(
        source,
        _request(
            "dataset-retokenized",
            mapping=FieldMapping("instruction", "response"),
            tokenizer=WordTokenizer("tokenizer-v2"),
        ),
    )
    repartitioned = service.import_json(
        source,
        _request(
            "dataset-repartitioned",
            mapping=FieldMapping("instruction", "response"),
            split_rule=SplitRule(99, (SplitPart("train", 1),)),
        ),
    )

    assert base.version.renderer_identity != remapped.version.renderer_identity
    assert base.version.tokenizer_identity != retokenized.version.tokenizer_identity
    assert base.version.split_identity != repartitioned.version.split_identity
    assert (
        len(
            {
                base.version.identity,
                remapped.version.identity,
                retokenized.version.identity,
                repartitioned.version.identity,
            }
        )
        == 4
    )
    assert renderer_identity(
        FieldMapping("instruction", "response"), RendererSpec()
    ) == (base.version.renderer_identity)


def test_correction_report_and_reimport_comparison_do_not_rewrite_versions(
    tmp_path: Path,
) -> None:
    service = DatasetService(tmp_path)
    previous = service.import_json(
        b'[{"instruction":"Alpha","response":"One"},{"instruction":"Beta"}]',
        _request(
            "dataset-before",
            mapping=FieldMapping("instruction", "response"),
            split_rule=SplitRule(1, (SplitPart("train", 1),)),
        ),
    )
    current = service.import_json(
        b'[{"instruction":"Alpha","response":"One"},{"instruction":"Gamma","response":"Three"}]',
        _request(
            "dataset-after",
            mapping=FieldMapping("instruction", "response"),
            split_rule=SplitRule(1, (SplitPart("validation", 1),)),
        ),
    )
    previous_payload = previous.version.to_payload()
    current_payload = current.version.to_payload()

    report = service.correction_report(previous.version)
    comparison = service.compare_reimport(previous.version, current.version)

    assert report.accepted_rows == 1
    assert [item.reason_code for item in report.exclusions] == ["missing_mapped_field"]
    assert report.reason_counts[0].count == 1
    assert len(comparison.added_content) == 1
    assert len(comparison.removed_content) == 0
    assert len(comparison.split_changes) == 1
    assert comparison.split_changes[0].previous_split == "train"
    assert comparison.split_changes[0].current_split == "validation"
    assert previous.version.to_payload() == previous_payload
    assert current.version.to_payload() == current_payload


class ChangingTokenizer(WordTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        self.calls += 1
        return self.calls


def test_nondeterministic_tokenizer_and_conflicting_version_fail_closed(
    tmp_path: Path,
) -> None:
    service = DatasetService(tmp_path)
    source = b'[{"instruction":"One","response":"First"}]'
    mapping = FieldMapping("instruction", "response")
    with pytest.raises(ApplicationServiceError, match="^tokenizer_nondeterministic$"):
        service.import_json(
            source,
            _request(
                "dataset-changing-tokenizer",
                mapping=mapping,
                tokenizer=ChangingTokenizer(),
            ),
        )

    service.import_json(source, _request("dataset-conflict", mapping=mapping))
    with pytest.raises(ApplicationServiceError, match="^dataset_version_conflict$"):
        service.import_json(
            b'[{"instruction":"Changed","response":"Second"}]',
            _request("dataset-conflict", mapping=mapping),
        )
