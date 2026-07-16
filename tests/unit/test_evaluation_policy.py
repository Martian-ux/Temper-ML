from dataclasses import replace
from decimal import Decimal
import hashlib

import pytest

from temper_ml.domain.evaluations import (
    ArtifactIntegrityStatus,
    ComparisonOperator,
    ConfidenceLabel,
    ConfidenceRule,
    EvaluationResult,
    EvaluatorKind,
    EvaluatorSpec,
    EvidenceStatus,
    HardQualifier,
    MetricDirection,
    MetricObservation,
    OptimizationObjective,
    RecommendationPolicy,
    SuiteEvidenceState,
    _pareto_results,
    build_recommendation,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import RecordReference, RecordValidationError
from temper_ml.domain.runs import EvaluationMode


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(record_type: str, logical_id: str) -> RecordReference:
    return RecordReference(
        record_type,
        logical_id,
        _identity(f"{record_type}:{logical_id}"),
    )


def _result(
    candidate_id: str,
    *,
    accuracy: int | Decimal,
    latency: int | Decimal,
    status: EvidenceStatus = EvidenceStatus.PASSED,
    latency_direction: MetricDirection = MetricDirection.MINIMIZE,
) -> EvaluationResult:
    return EvaluationResult(
        result_id=f"result-{candidate_id}",
        candidate=_reference("artifact", candidate_id),
        evaluation_mode=EvaluationMode.FULL_SUITE,
        artifact_integrity_status=ArtifactIntegrityStatus.PASSED,
        artifact_integrity_evidence=_identity(f"integrity:{candidate_id}"),
        evidence_status=status,
        metrics=(
            MetricObservation(
                "accuracy",
                EvaluatorKind.TASK_METRIC,
                accuracy,
                MetricDirection.MAXIMIZE,
            ),
            MetricObservation(
                "latency",
                EvaluatorKind.DETERMINISTIC_CHECK,
                latency,
                latency_direction,
            ),
        ),
        suite=_reference("evaluation_suite", "suite-confirmation"),
        suite_state=SuiteEvidenceState.SEALED,
    )


def _policy() -> RecommendationPolicy:
    return RecommendationPolicy(
        policy_id="policy-selection",
        hard_qualifiers=(
            HardQualifier(
                "accuracy",
                ComparisonOperator.GREATER_THAN_OR_EQUAL,
                8,
            ),
        ),
        advisory_metrics=("format_validity",),
        objectives=(
            OptimizationObjective("accuracy", MetricDirection.MAXIMIZE),
            OptimizationObjective("latency", MetricDirection.MINIMIZE),
        ),
        baseline_comparisons=(),
        confidence_rules=(
            ConfidenceRule(
                ConfidenceLabel.HIGH,
                (EvidenceStatus.PASSED,),
                (SuiteEvidenceState.SEALED,),
                minimum_metric_count=2,
            ),
            ConfidenceRule(
                ConfidenceLabel.LOW,
                (EvidenceStatus.SUBJECTIVE_ONLY,),
            ),
        ),
    )


@pytest.mark.parametrize(
    "kind",
    (
        EvaluatorKind.DETERMINISTIC_CHECK,
        EvaluatorKind.HELD_OUT_LOSS,
        EvaluatorKind.TASK_METRIC,
        EvaluatorKind.FORMAT_CHECK,
    ),
)
def test_v1_evaluator_contract_accepts_only_explainable_local_kinds(
    kind: EvaluatorKind,
) -> None:
    assert (
        EvaluatorSpec(
            f"evaluator-{kind.value}",
            kind,
            f"metric-{kind.value}",
            MetricDirection.MAXIMIZE,
        ).kind
        is kind
    )


def test_v1_evaluator_contract_rejects_model_judges_and_universal_scores() -> None:
    with pytest.raises(RecordValidationError, match="model-judge"):
        EvaluatorSpec(
            "evaluator-model-judge",
            EvaluatorKind.MODEL_JUDGE,
            "preference",
            MetricDirection.MAXIMIZE,
        )
    with pytest.raises(RecordValidationError, match="aggregate"):
        EvaluatorSpec(
            "evaluator-aggregate",
            EvaluatorKind.TASK_METRIC,
            "universal_quality_score",
            MetricDirection.MAXIMIZE,
        )


def test_policy_engine_qualifies_then_ranks_lexicographically_and_exposes_pareto() -> (
    None
):
    primary = _result("artifact-primary", accuracy=9, latency=9)
    tradeoff = _result("artifact-tradeoff", accuracy=8, latency=1)
    failed = _result("artifact-failed", accuracy=7, latency=1)

    recommendation = build_recommendation(
        "recommendation-selection",
        _policy(),
        (failed, tradeoff, primary),
    )

    assert recommendation.selected_candidate == primary.candidate
    assert recommendation.pareto_alternatives == (tradeoff.candidate,)
    assert recommendation.confidence is ConfidenceLabel.HIGH
    assert "qualified_objective_tradeoff" in recommendation.conflicts
    assessments = {
        item.candidate.logical_id: item for item in recommendation.assessments
    }
    assert assessments["artifact-primary"].rank == 1
    assert assessments["artifact-tradeoff"].rank == 2
    assert assessments["artifact-failed"].rank is None
    assert assessments["artifact-failed"].qualifier_failures == (
        "failed_qualifier:accuracy",
    )
    assert not hasattr(recommendation, "score")


def test_policy_engine_uses_identity_tie_breaking_and_preserves_evidence_status() -> (
    None
):
    first = _result("artifact-a", accuracy=9, latency=2)
    second = _result("artifact-b", accuracy=9, latency=2)

    forward = build_recommendation("recommendation-tie", _policy(), (second, first))
    reverse = build_recommendation("recommendation-tie", _policy(), (first, second))

    assert forward.identity == reverse.identity
    assert forward.selected_candidate == first.candidate
    assert [item.evidence_status for item in forward.assessments] == [
        EvidenceStatus.PASSED,
        EvidenceStatus.PASSED,
    ]
    reordered = replace(forward, assessments=tuple(reversed(forward.assessments)))
    assert reordered.assessments == forward.assessments
    assert reordered.identity == forward.identity


def test_policy_engine_uses_transitive_best_anchored_tolerance_cohorts() -> None:
    best = _result("artifact-z-best", accuracy=10, latency=1)
    middle = _result("artifact-y-middle", accuracy=Decimal("9.5"), latency=1)
    chained = _result("artifact-a-chained", accuracy=9, latency=1)
    policy = replace(
        _policy(),
        hard_qualifiers=(),
        advisory_metrics=(),
        objectives=(
            OptimizationObjective(
                "accuracy",
                MetricDirection.MAXIMIZE,
                Decimal("0.6"),
            ),
        ),
    )

    recommendation = build_recommendation(
        "recommendation-chained-tolerance",
        policy,
        (chained, middle, best),
    )
    ranks = {
        assessment.candidate.logical_id: assessment.rank
        for assessment in recommendation.assessments
    }

    assert recommendation.selected_candidate == middle.candidate
    assert ranks == {
        "artifact-a-chained": 3,
        "artifact-y-middle": 1,
        "artifact-z-best": 2,
    }
    assert chained.candidate not in recommendation.pareto_alternatives


def test_selected_candidate_is_pareto_nondominated_under_global_cohorts() -> None:
    candidate_a = _result(
        "artifact-a",
        accuracy=10,
        latency=0,
        latency_direction=MetricDirection.MAXIMIZE,
    )
    candidate_b = _result(
        "artifact-b",
        accuracy=9,
        latency=5,
        latency_direction=MetricDirection.MAXIMIZE,
    )
    candidate_c = _result(
        "artifact-c",
        accuracy=8,
        latency=10,
        latency_direction=MetricDirection.MAXIMIZE,
    )
    policy = replace(
        _policy(),
        hard_qualifiers=(),
        advisory_metrics=(),
        objectives=(
            OptimizationObjective("accuracy", MetricDirection.MAXIMIZE, 1),
            OptimizationObjective("latency", MetricDirection.MAXIMIZE, 0),
        ),
    )
    results = (candidate_a, candidate_b, candidate_c)

    recommendation = build_recommendation(
        "recommendation-global-cohort-pareto",
        policy,
        results,
    )
    pareto_front = {result.candidate for result in _pareto_results(results, policy)}

    assert recommendation.selected_candidate == candidate_b.candidate
    assert pareto_front == {candidate_b.candidate, candidate_c.candidate}
    assert recommendation.selected_candidate in pareto_front
    assert recommendation.pareto_alternatives == (candidate_c.candidate,)
    assert recommendation.confidence is ConfidenceLabel.HIGH
    assert "qualified_objective_tradeoff" in recommendation.conflicts


def test_policy_engine_rejects_direction_and_cross_candidate_semantic_conflicts() -> (
    None
):
    first = _result("artifact-direction-a", accuracy=9, latency=2)
    second = _result("artifact-direction-b", accuracy=8, latency=1)
    reversed_policy = replace(
        _policy(),
        objectives=(
            OptimizationObjective("accuracy", MetricDirection.MINIMIZE),
            OptimizationObjective("latency", MetricDirection.MINIMIZE),
        ),
    )
    conflicting_second = replace(
        second,
        metrics=(
            replace(second.metrics[0], direction=MetricDirection.MINIMIZE),
            second.metrics[1],
        ),
    )

    with pytest.raises(RecordValidationError, match="objective direction conflicts"):
        build_recommendation(
            "recommendation-direction-conflict",
            reversed_policy,
            (first,),
        )
    with pytest.raises(RecordValidationError, match="metric semantics conflict"):
        build_recommendation(
            "recommendation-semantics-conflict",
            _policy(),
            (first, conflicting_second),
        )


def test_policy_engine_rejects_different_suite_content_identities() -> None:
    first = _result("artifact-suite-a", accuracy=9, latency=2)
    second = replace(
        _result("artifact-suite-b", accuracy=9, latency=2),
        suite=RecordReference(
            "evaluation_suite",
            "suite-confirmation",
            _identity("different-suite-content"),
        ),
    )

    with pytest.raises(RecordValidationError, match="share one evaluation cohort"):
        build_recommendation(
            "recommendation-suite-identity-conflict",
            _policy(),
            (first, second),
        )


def test_policy_engine_rejects_different_suite_states() -> None:
    sealed = _result("artifact-suite-sealed", accuracy=9, latency=2)
    modified = replace(
        _result("artifact-suite-modified", accuracy=9, latency=2),
        suite_state=SuiteEvidenceState.MODIFIED,
    )

    with pytest.raises(RecordValidationError, match="share one evaluation cohort"):
        build_recommendation(
            "recommendation-suite-state-conflict",
            _policy(),
            (sealed, modified),
        )


def test_policy_engine_rejects_light_and_full_suite_results() -> None:
    full = _result("artifact-full-suite", accuracy=9, latency=2)
    light = replace(
        _result("artifact-light", accuracy=9, latency=2),
        evaluation_mode=EvaluationMode.LIGHT_EVALUATION,
        suite=None,
        suite_state=None,
    )

    with pytest.raises(RecordValidationError, match="share one evaluation cohort"):
        build_recommendation(
            "recommendation-mode-suite-conflict",
            _policy(),
            (full, light),
        )


def test_policy_engine_rejects_different_modes_on_one_development_suite() -> None:
    development_suite = _reference("evaluation_suite", "suite-development")
    full = replace(
        _result("artifact-development-full", accuracy=9, latency=2),
        suite=development_suite,
    )
    experiment = replace(
        _result("artifact-development-loop", accuracy=9, latency=2),
        evaluation_mode=EvaluationMode.EXPERIMENT_LOOP,
        suite=development_suite,
    )

    with pytest.raises(RecordValidationError, match="share one evaluation cohort"):
        build_recommendation(
            "recommendation-suite-mode-conflict",
            _policy(),
            (full, experiment),
        )


def test_policy_engine_discloses_mixed_context_non_rankable_evidence() -> None:
    selected = _result("artifact-selected", accuracy=9, latency=2)
    non_rankable = replace(
        _result("artifact-non-rankable", accuracy=9, latency=2),
        evaluation_mode=EvaluationMode.LIGHT_EVALUATION,
        evidence_status=EvidenceStatus.INCONCLUSIVE,
        suite=None,
        suite_state=None,
    )

    recommendation = build_recommendation(
        "recommendation-mixed-context-disclosure",
        _policy(),
        (non_rankable, selected),
    )

    assert recommendation.selected_candidate == selected.candidate
    assert tuple(item.candidate for item in recommendation.assessments) == (
        non_rankable.candidate,
        selected.candidate,
    )
    assert recommendation.assessments[0].qualifier_failures == (
        "evidence_status:inconclusive",
    )


def test_no_quality_result_keeps_integrity_separate_and_quality_unevaluated() -> None:
    result = EvaluationResult(
        result_id="result-no-quality",
        candidate=_reference("artifact", "artifact-no-quality"),
        evaluation_mode=EvaluationMode.NO_QUALITY_EVALUATION,
        artifact_integrity_status=ArtifactIntegrityStatus.PASSED,
        artifact_integrity_evidence=_identity("integrity-no-quality"),
        evidence_status=EvidenceStatus.UNEVALUATED,
    )

    assert result.artifact_integrity_status is ArtifactIntegrityStatus.PASSED
    assert result.evidence_status is EvidenceStatus.UNEVALUATED
    assert result.metrics == ()


def test_failed_integrity_stays_separate_but_cannot_be_policy_qualified() -> None:
    result = replace(
        _result("artifact-integrity-failed", accuracy=9, latency=1),
        artifact_integrity_status=ArtifactIntegrityStatus.FAILED,
    )

    recommendation = build_recommendation(
        "recommendation-integrity-failed",
        _policy(),
        (result,),
    )

    assert recommendation.selected_candidate is None
    assert recommendation.assessments[0].evidence_status is EvidenceStatus.PASSED
    assert recommendation.assessments[0].qualifier_failures == (
        "artifact_integrity:failed",
    )
