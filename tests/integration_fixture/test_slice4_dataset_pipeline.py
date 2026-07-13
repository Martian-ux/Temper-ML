import hashlib
import json
from pathlib import Path

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
from temper_ml.domain.records import RecordEnvelope
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore


class FixtureTokenizer:
    identity = ContentIdentity(
        "sha256", hashlib.sha256(b"fixture-tokenizer-v1").hexdigest()
    )

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.encode("utf-8"))


def _request(version_id: str, seed: int) -> DatasetImportRequest:
    return DatasetImportRequest(
        version_id=version_id,
        field_mapping=FieldMapping("instruction", "response", "context"),
        renderer=RendererSpec(),
        filter_rule=FilterRule(1, 500, 500),
        deduplication_rule=DeduplicationRule(),
        split_rule=SplitRule(
            seed,
            (SplitPart("train", 4), SplitPart("validation", 1)),
        ),
        tokenizer=FixtureTokenizer(),
        preview_limit=2,
    )


def test_slice_four_forms_one_recoverable_deterministic_local_workflow(
    tmp_path: Path,
) -> None:
    initial_rows = [
        {
            "instruction": "Summarize the synthetic note",
            "context": "Alpha fixture context",
            "response": "Alpha fixture summary",
        },
        {
            "instruction": "Rewrite the synthetic note",
            "context": "Beta fixture context",
            "response": "Beta fixture rewrite",
        },
        {
            "instruction": "Summarize the synthetic note",
            "context": "Alpha fixture context",
            "response": "Alpha fixture summary",
        },
        {"instruction": "Incomplete synthetic row", "context": "No answer"},
    ]
    corrected_rows = [
        initial_rows[0],
        initial_rows[1],
        {
            "instruction": "Classify the synthetic note",
            "context": "Gamma fixture context",
            "response": "Gamma fixture label",
        },
        {
            "instruction": "Complete the corrected row",
            "context": "Corrected fixture context",
            "response": "Corrected fixture answer",
        },
    ]
    service = DatasetService(tmp_path)

    initial = service.import_json(
        json.dumps(initial_rows, separators=(",", ":")).encode(),
        _request("dataset-fixture-v1", 71),
    )
    repeated = service.import_json(
        json.dumps(initial_rows, separators=(",", ":")).encode(),
        _request("dataset-fixture-v1", 71),
    )
    corrected = service.import_hugging_face_rows(
        corrected_rows,
        _request("dataset-fixture-v2", 71),
    )

    assert initial.rendered_bytes == repeated.rendered_bytes
    assert initial.version.identity == repeated.version.identity
    assert initial.version.split_membership == repeated.version.split_membership
    assert initial.version.statistics.accepted_rows == 2
    assert initial.version.statistics.excluded_rows == 2
    assert corrected.version.statistics.accepted_rows == 4
    report = service.correction_report(initial.version)
    assert [item.reason_code for item in report.exclusions] == [
        "duplicate_rendered_text",
        "missing_mapped_field",
    ]
    comparison = service.compare_reimport(initial.version, corrected.version)
    assert len(comparison.added_content) == 2
    assert comparison.removed_content == ()

    reopened = TypedEvidenceStore(tmp_path)
    verification = reopened.verify()
    assert verification.record_counts == {"dataset_version": 2}
    assert verification.event_count == 0
    stored = [
        item
        for item in reopened.iter_records()
        if item.reference.logical_id == "dataset-fixture-v1"
    ]
    assert len(stored) == 1
    decoded = RecordEnvelope.from_dict(stored[0].envelope.to_dict()).to_record()
    assert decoded == initial.version
    assert dumps_canonical_json(decoded.to_dict()) == dumps_canonical_json(
        initial.version.to_dict()
    )
    public_bytes = dumps_canonical_json(reopened.public_dump().value)
    for row in initial_rows + corrected_rows:
        for value in row.values():
            assert value.encode() not in public_bytes
