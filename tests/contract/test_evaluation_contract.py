from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from temper_ml.domain.evaluations import (
    ArtifactIntegrityStatus,
    CaseSuiteKind,
    ComparisonOperator,
    ConfidenceLabel,
    ConfidenceRule,
    EvaluationCase,
    EvaluationResult,
    EvaluationSuite,
    EvaluatorKind,
    EvaluatorSpec,
    EvidenceStatus,
    HardQualifier,
    MetricDirection,
    MetricObservation,
    OptimizationObjective,
    RecommendationPolicy,
    Review,
    ReviewCandidate,
    ReviewEntry,
    ReviewMode,
    ReviewOutput,
    ReviewRating,
    ReviewStage,
    SuiteEvidenceState,
    UserDecision,
    UserDecisionStatus,
    build_recommendation,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordEnvelope,
    RecordReference,
    RecordValidationError,
    record_reference,
)
from temper_ml.domain.runs import EvaluationMode

REPO_ROOT = Path(__file__).parents[2]


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(record_type: str, logical_id: str) -> RecordReference:
    return RecordReference(
        record_type, logical_id, _identity(f"{record_type}:{logical_id}")
    )


def test_slice_six_record_graph_round_trips_through_strict_envelopes() -> None:
    candidate = _reference("artifact", "artifact-contract")
    initial_suite = EvaluationSuite(
        "suite-contract",
        CaseSuiteKind.CONFIRMATION,
        SuiteEvidenceState.UNSEALED,
        (EvaluationCase("case-contract", _identity("case-contract")),),
        (
            EvaluatorSpec(
                "evaluator-contract",
                EvaluatorKind.TASK_METRIC,
                "accuracy",
                MetricDirection.MAXIMIZE,
            ),
        ),
    )
    suite = replace(
        initial_suite,
        state=SuiteEvidenceState.SEALED,
        prior_suite=record_reference(initial_suite),
    )
    review = Review(
        "review-contract",
        ReviewMode.SOLO,
        ReviewStage.RECORDED,
        (
            ReviewEntry(
                "prompt-contract",
                {"text": "Synthetic contract prompt"},
                {"temperature": 0},
                (ReviewOutput("candidate-001", {"text": "Synthetic output"}),),
                "The synthetic response followed the requested format.",
                (ReviewRating("candidate-001", "format", 1),),
            ),
        ),
        "I reviewed all synthetic outputs under the recorded settings.",
        (ReviewCandidate("candidate-001", candidate),),
        False,
    )
    with pytest.raises(RecordValidationError, match="solo reviews"):
        replace(review, leak_audit_passed=True)
    result = EvaluationResult(
        "result-contract",
        candidate,
        EvaluationMode.FULL_SUITE,
        ArtifactIntegrityStatus.PASSED,
        _identity("artifact-integrity-contract"),
        EvidenceStatus.PASSED,
        metrics=(
            MetricObservation(
                "accuracy",
                EvaluatorKind.TASK_METRIC,
                9,
                MetricDirection.MAXIMIZE,
            ),
        ),
        suite=record_reference(suite),
        suite_state=SuiteEvidenceState.SEALED,
        review=record_reference(review),
    )
    policy = RecommendationPolicy(
        "recommendation-policy-contract",
        (
            HardQualifier(
                "accuracy",
                ComparisonOperator.GREATER_THAN_OR_EQUAL,
                8,
            ),
        ),
        (),
        (OptimizationObjective("accuracy", MetricDirection.MAXIMIZE),),
        (),
        (
            ConfidenceRule(
                ConfidenceLabel.HIGH,
                (EvidenceStatus.PASSED,),
                (SuiteEvidenceState.SEALED,),
                1,
            ),
        ),
    )
    recommendation = build_recommendation(
        "recommendation-contract",
        policy,
        (result,),
    )
    decision = UserDecision(
        "decision-contract",
        record_reference(recommendation),
        candidate,
        UserDecisionStatus.SELECTED,
        EvidenceStatus.PASSED,
    )

    for record in (suite, review, result, policy, recommendation, decision):
        envelope = record.to_envelope()
        decoded = RecordEnvelope.from_dict(envelope.to_dict()).to_record()
        assert type(decoded) is type(record)
        assert decoded == record
        assert decoded.identity == record.identity


def test_published_evaluation_schemas_are_bounded_and_reject_model_judges() -> None:
    records = REPO_ROOT / "schemas" / "records"
    suite = json.loads(
        (records / "evaluation_suite.schema.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (records / "evaluation_result.schema.json").read_text(encoding="utf-8")
    )
    recommendation = json.loads(
        (records / "recommendation.schema.json").read_text(encoding="utf-8")
    )

    evaluator_kinds = suite["$defs"]["evaluator"]["properties"]["kind"]["enum"]
    metric_kinds = result["$defs"]["metric"]["properties"]["evaluator_kind"]["enum"]
    assert evaluator_kinds == [
        "deterministic_check",
        "held_out_loss",
        "task_metric",
        "format_check",
    ]
    assert metric_kinds == evaluator_kinds
    assert "model_judge" not in evaluator_kinds
    assert (
        "score" not in recommendation["allOf"][1]["properties"]["payload"]["properties"]
    )


def test_evaluation_mode_schema_covers_all_adopted_modes() -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas" / "records" / "evaluation_result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    modes = schema["allOf"][1]["properties"]["payload"]["properties"][
        "evaluation_mode"
    ]["enum"]

    assert modes == [
        EvaluationMode.NO_QUALITY_EVALUATION.value,
        EvaluationMode.LIGHT_EVALUATION.value,
        EvaluationMode.FULL_SUITE.value,
        EvaluationMode.EXPERIMENT_LOOP.value,
    ]


def test_evaluation_modes_enforce_their_suite_evidence_contract() -> None:
    candidate = _reference("artifact", "artifact-mode-contract")
    suite = _reference("evaluation_suite", "suite-mode-contract")
    metric = MetricObservation(
        "accuracy",
        EvaluatorKind.TASK_METRIC,
        1,
        MetricDirection.MAXIMIZE,
    )
    common = {
        "candidate": candidate,
        "artifact_integrity_status": ArtifactIntegrityStatus.PASSED,
        "artifact_integrity_evidence": _identity("mode-contract-integrity"),
    }

    for mode in (EvaluationMode.FULL_SUITE, EvaluationMode.EXPERIMENT_LOOP):
        with pytest.raises(RecordValidationError, match="require suite evidence"):
            EvaluationResult(
                result_id=f"result-{mode.value}-missing-suite",
                evaluation_mode=mode,
                evidence_status=EvidenceStatus.PASSED,
                metrics=(metric,),
                **common,
            )
        assert (
            EvaluationResult(
                result_id=f"result-{mode.value}-suite-backed",
                evaluation_mode=mode,
                evidence_status=EvidenceStatus.PASSED,
                metrics=(metric,),
                suite=suite,
                suite_state=SuiteEvidenceState.SEALED,
                **common,
            ).suite
            == suite
        )

    assert (
        EvaluationResult(
            result_id="result-light-without-suite",
            evaluation_mode=EvaluationMode.LIGHT_EVALUATION,
            evidence_status=EvidenceStatus.PASSED,
            metrics=(metric,),
            **common,
        ).suite
        is None
    )
    with pytest.raises(
        RecordValidationError, match="must not reference suite evidence"
    ):
        EvaluationResult(
            result_id="result-light-claiming-suite",
            evaluation_mode=EvaluationMode.LIGHT_EVALUATION,
            evidence_status=EvidenceStatus.PASSED,
            metrics=(metric,),
            suite=suite,
            suite_state=SuiteEvidenceState.SEALED,
            **common,
        )
    with pytest.raises(
        RecordValidationError, match="must not reference suite evidence"
    ):
        EvaluationResult(
            result_id="result-no-quality-claiming-suite",
            evaluation_mode=EvaluationMode.NO_QUALITY_EVALUATION,
            evidence_status=EvidenceStatus.UNEVALUATED,
            suite=suite,
            suite_state=SuiteEvidenceState.SEALED,
            **common,
        )
