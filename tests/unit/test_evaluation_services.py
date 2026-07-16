from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.evaluations import EvaluationService
from temper_ml.cli import main
from temper_ml.domain.artifacts import Artifact
from temper_ml.domain.evaluations import (
    ArtifactIntegrityStatus,
    BaselineComparison,
    BaselineOutcome,
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
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    record_reference,
)
from temper_ml.domain.runs import EvaluationMode
from temper_ml.store.evidence import TypedEvidenceStore


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _foundation(root: Path, capsys) -> tuple[Artifact, Artifact, Artifact]:
    assert main(["fixture-workflow", str(root)]) == 0
    capsys.readouterr()
    store = TypedEvidenceStore(root)
    first = next(
        stored.record
        for stored in store.iter_records()
        if isinstance(stored.record, Artifact)
    )
    second = replace(first, artifact_id="artifact-evaluation-alternative")
    baseline = replace(first, artifact_id="artifact-evaluation-baseline")
    store.write_record(second)
    store.write_record(baseline)
    store.verify()
    return first, second, baseline


def _suite() -> EvaluationSuite:
    return EvaluationSuite(
        "suite-evaluation-service",
        CaseSuiteKind.CONFIRMATION,
        SuiteEvidenceState.UNSEALED,
        (
            EvaluationCase("case-one", _identity("evaluation-case-one")),
            EvaluationCase("case-two", _identity("evaluation-case-two")),
        ),
        (
            EvaluatorSpec(
                "accuracy-evaluator",
                EvaluatorKind.TASK_METRIC,
                "accuracy",
                MetricDirection.MAXIMIZE,
            ),
            EvaluatorSpec(
                "latency-check",
                EvaluatorKind.DETERMINISTIC_CHECK,
                "latency",
                MetricDirection.MINIMIZE,
            ),
        ),
    )


def _result(
    result_id: str,
    artifact: Artifact,
    suite: EvaluationSuite,
    *,
    accuracy: int,
    latency: int,
    status: EvidenceStatus = EvidenceStatus.PASSED,
    baseline_result: EvaluationResult | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        result_id=result_id,
        candidate=record_reference(artifact),
        evaluation_mode=EvaluationMode.FULL_SUITE,
        artifact_integrity_status=ArtifactIntegrityStatus.PASSED,
        artifact_integrity_evidence=artifact.integrity_evidence,
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
                MetricDirection.MINIMIZE,
            ),
        ),
        baseline_comparisons=(
            (
                BaselineComparison(
                    "fixture-baseline",
                    "accuracy",
                    record_reference(baseline_result),
                    accuracy,
                    7,
                    (
                        BaselineOutcome.BETTER
                        if accuracy > 7
                        else BaselineOutcome.EQUIVALENT
                    ),
                ),
            )
            if baseline_result is not None
            else ()
        ),
        suite=record_reference(suite),
        suite_state=suite.state,
    )


def _policy() -> RecommendationPolicy:
    return RecommendationPolicy(
        "policy-evaluation-service",
        (
            HardQualifier(
                "accuracy",
                ComparisonOperator.GREATER_THAN_OR_EQUAL,
                8,
            ),
        ),
        ("format_validity",),
        (
            OptimizationObjective("accuracy", MetricDirection.MAXIMIZE),
            OptimizationObjective("latency", MetricDirection.MINIMIZE),
        ),
        ("fixture-baseline",),
        (
            ConfidenceRule(
                ConfidenceLabel.HIGH,
                (EvidenceStatus.PASSED,),
                (SuiteEvidenceState.SEALED,),
                2,
            ),
        ),
    )


def test_services_persist_rank_recommend_and_keep_warned_decisions_separate(
    tmp_path: Path,
    capsys,
) -> None:
    first, second, baseline_artifact = _foundation(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    sealed = service.seal_suite(service.register_suite(_suite()))
    baseline = service.record_result(
        _result(
            "result-baseline",
            baseline_artifact,
            sealed,
            accuracy=7,
            latency=5,
        )
    )
    preferred = service.record_result(
        _result(
            "result-preferred",
            first,
            sealed,
            accuracy=9,
            latency=8,
            baseline_result=baseline,
        )
    )
    warned = service.record_result(
        _result(
            "result-warned",
            second,
            sealed,
            accuracy=7,
            latency=1,
            status=EvidenceStatus.INCONCLUSIVE,
            baseline_result=baseline,
        )
    )
    policy = service.register_policy(_policy())
    recommendation = service.recommend(
        "recommendation-service",
        policy,
        (warned, preferred),
    )
    before = recommendation.to_dict()
    decision = service.record_decision(
        UserDecision(
            "decision-warned-selection",
            record_reference(recommendation),
            warned.candidate,
            UserDecisionStatus.SELECTED,
            EvidenceStatus.INCONCLUSIVE,
            "Selected for a bounded synthetic follow-up despite incomplete evidence.",
        )
    )

    assert recommendation.selected_candidate == preferred.candidate
    assert recommendation.confidence is ConfidenceLabel.HIGH
    assert "advisory_metric_missing:format_validity" in recommendation.conflicts
    assert decision.evidence_status_at_decision is EvidenceStatus.INCONCLUSIVE
    assert recommendation.to_dict() == before
    stored_recommendation = service.store.read_record(
        record_reference(recommendation)
    ).record
    stored_warned = service.store.read_record(record_reference(warned)).record
    assert stored_recommendation.to_dict() == before
    assert isinstance(stored_warned, EvaluationResult)
    assert stored_warned.evidence_status is EvidenceStatus.INCONCLUSIVE


def test_confirmation_inspection_invalidates_old_results_for_new_recommendations(
    tmp_path: Path,
    capsys,
) -> None:
    first, _, baseline_artifact = _foundation(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    sealed = service.seal_suite(service.register_suite(_suite()))
    baseline = service.record_result(
        _result(
            "result-inspection-baseline",
            baseline_artifact,
            sealed,
            accuracy=7,
            latency=5,
        )
    )
    result = service.record_result(
        _result(
            "result-before-inspection",
            first,
            sealed,
            accuracy=9,
            latency=2,
            baseline_result=baseline,
        )
    )
    policy = service.register_policy(_policy())
    inspected = service.inspect_confirmation_suite(sealed)

    with pytest.raises(ApplicationServiceError, match="revision_stale"):
        service.recommend("recommendation-stale", policy, (result,))
    with pytest.raises(ApplicationServiceError, match="contaminated"):
        service.record_result(
            _result(
                "result-after-inspection-invalid",
                first,
                inspected,
                accuracy=9,
                latency=2,
                baseline_result=baseline,
            )
        )
    contaminated = service.record_result(
        _result(
            "result-after-inspection",
            first,
            inspected,
            accuracy=9,
            latency=2,
            status=EvidenceStatus.CONTAMINATED,
        )
    )
    assert contaminated.suite_state is SuiteEvidenceState.UNSEALED
    assert contaminated.evidence_status is EvidenceStatus.CONTAMINATED


def test_service_rejects_confirmation_suite_for_experiment_loop_results(
    tmp_path: Path,
    capsys,
) -> None:
    first, _, _ = _foundation(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    confirmation = service.seal_suite(service.register_suite(_suite()))
    confirmation_result = replace(
        _result(
            "result-experiment-confirmation-invalid",
            first,
            confirmation,
            accuracy=9,
            latency=2,
        ),
        evaluation_mode=EvaluationMode.EXPERIMENT_LOOP,
    )

    with pytest.raises(ApplicationServiceError, match="confirmation_suite_invalid"):
        service.record_result(confirmation_result)

    development = service.seal_suite(
        service.register_suite(
            replace(
                _suite(),
                suite_id="suite-experiment-development",
                kind=CaseSuiteKind.DEVELOPMENT,
            )
        )
    )
    accepted = service.record_result(
        replace(
            _result(
                "result-experiment-development",
                first,
                development,
                accuracy=9,
                latency=2,
            ),
            evaluation_mode=EvaluationMode.EXPERIMENT_LOOP,
        )
    )
    assert accepted.evaluation_mode is EvaluationMode.EXPERIMENT_LOOP
    assert accepted.suite == record_reference(development)


def test_service_rejects_undeclared_metrics_and_conflicting_logical_revisions(
    tmp_path: Path,
    capsys,
) -> None:
    first, second, baseline_artifact = _foundation(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    sealed = service.seal_suite(service.register_suite(_suite()))
    baseline = service.record_result(
        _result(
            "result-conflict-baseline",
            baseline_artifact,
            sealed,
            accuracy=7,
            latency=5,
        )
    )
    original = service.record_result(
        _result(
            "result-conflict",
            first,
            sealed,
            accuracy=9,
            latency=2,
            baseline_result=baseline,
        )
    )
    conflicting = _result(
        "result-conflict",
        second,
        sealed,
        accuracy=8,
        latency=1,
        baseline_result=baseline,
    )
    undeclared = replace(
        _result(
            "result-undeclared",
            second,
            sealed,
            accuracy=8,
            latency=1,
            baseline_result=baseline,
        ),
        metrics=(
            MetricObservation(
                "unknown_metric",
                EvaluatorKind.TASK_METRIC,
                1,
                MetricDirection.MAXIMIZE,
            ),
        ),
    )

    with pytest.raises(ApplicationServiceError, match="not_declared"):
        service.record_result(undeclared)
    with pytest.raises(ApplicationServiceError, match="revision_conflict"):
        service.record_result(conflicting)
    assert service.store.read_record(record_reference(original)).record == original


def test_result_admission_binds_integrity_and_baselines_to_exact_evidence(
    tmp_path: Path,
    capsys,
) -> None:
    first, _, baseline_artifact = _foundation(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    sealed = service.seal_suite(service.register_suite(_suite()))
    baseline = service.record_result(
        _result(
            "result-binding-baseline",
            baseline_artifact,
            sealed,
            accuracy=7,
            latency=5,
        )
    )
    valid = _result(
        "result-binding-valid",
        first,
        sealed,
        accuracy=9,
        latency=2,
        baseline_result=baseline,
    )
    comparison = valid.baseline_comparisons[0]

    with pytest.raises(RecordValidationError, match="reference evaluation_result"):
        BaselineComparison(
            "artifact-is-not-baseline-evidence",
            "accuracy",
            record_reference(first),
            9,
            7,
            BaselineOutcome.BETTER,
        )
    with pytest.raises(ApplicationServiceError, match="integrity_evidence_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-wrong-integrity",
                artifact_integrity_evidence=_identity("wrong-integrity-evidence"),
            )
        )
    with pytest.raises(ApplicationServiceError, match="candidate_value_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-wrong-candidate-value",
                baseline_comparisons=(replace(comparison, candidate_value=8),),
            )
        )
    with pytest.raises(ApplicationServiceError, match="baseline_value_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-wrong-baseline-value",
                baseline_comparisons=(replace(comparison, baseline_value=6),),
            )
        )
    with pytest.raises(ApplicationServiceError, match="baseline_outcome_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-wrong-outcome",
                baseline_comparisons=(
                    replace(comparison, outcome=BaselineOutcome.WORSE),
                ),
            )
        )
    with pytest.raises(ApplicationServiceError, match="self_reference"):
        service.record_result(
            _result(
                "result-binding-self-baseline",
                baseline_artifact,
                sealed,
                accuracy=7,
                latency=5,
                baseline_result=baseline,
            )
        )

    rogue_baseline = replace(
        baseline,
        result_id="result-binding-direction-baseline",
        metrics=(
            replace(baseline.metrics[0], direction=MetricDirection.MINIMIZE),
            baseline.metrics[1],
        ),
    )
    service.store.write_record(rogue_baseline)
    with pytest.raises(ApplicationServiceError, match="semantics_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-wrong-semantics",
                baseline_comparisons=(
                    replace(comparison, baseline=record_reference(rogue_baseline)),
                ),
            )
        )

    other_suite = service.seal_suite(
        service.register_suite(
            replace(_suite(), suite_id="suite-binding-other-context")
        )
    )
    other_suite_baseline = service.record_result(
        _result(
            "result-binding-other-suite-baseline",
            baseline_artifact,
            other_suite,
            accuracy=7,
            latency=5,
        )
    )
    with pytest.raises(ApplicationServiceError, match="suite_context_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-cross-suite",
                baseline_comparisons=(
                    replace(
                        comparison,
                        baseline=record_reference(other_suite_baseline),
                    ),
                ),
            )
        )

    cross_mode_baseline = replace(
        baseline,
        result_id="result-binding-cross-mode-baseline",
        evaluation_mode=EvaluationMode.EXPERIMENT_LOOP,
    )
    service.store.write_record(cross_mode_baseline)
    with pytest.raises(ApplicationServiceError, match="evaluation_mode_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-cross-mode",
                baseline_comparisons=(
                    replace(
                        comparison,
                        baseline=record_reference(cross_mode_baseline),
                    ),
                ),
            )
        )

    wrong_state_baseline = replace(
        baseline,
        result_id="result-binding-wrong-state-baseline",
        suite_state=SuiteEvidenceState.UNSEALED,
    )
    service.store.write_record(wrong_state_baseline)
    with pytest.raises(ApplicationServiceError, match="suite_context_mismatch"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-cross-state",
                baseline_comparisons=(
                    replace(
                        comparison,
                        baseline=record_reference(wrong_state_baseline),
                    ),
                ),
            )
        )

    subjective_review = service.record_solo_review(
        Review(
            "review-binding-subjective",
            ReviewMode.SOLO,
            ReviewStage.RECORDED,
            (
                ReviewEntry(
                    "prompt-binding-subjective",
                    {"text": "Inspect the synthetic baseline."},
                    {"temperature": 0},
                    (
                        ReviewOutput(
                            "candidate-001",
                            {"text": "Synthetic baseline output"},
                        ),
                    ),
                    "The synthetic output was inspected.",
                    (ReviewRating("candidate-001", "task-fit", 1),),
                ),
            ),
            "I inspected the synthetic baseline output.",
            (
                ReviewCandidate(
                    "candidate-001",
                    record_reference(baseline_artifact),
                ),
            ),
            False,
        )
    )
    for invalid_status in (
        EvidenceStatus.CONTAMINATED,
        EvidenceStatus.INCONCLUSIVE,
        EvidenceStatus.SUBJECTIVE_ONLY,
        EvidenceStatus.UNEVALUATED,
    ):
        invalid_status_baseline = replace(
            baseline,
            result_id=f"result-binding-{invalid_status.value}-baseline",
            evidence_status=invalid_status,
            review=(
                record_reference(subjective_review)
                if invalid_status is EvidenceStatus.SUBJECTIVE_ONLY
                else None
            ),
        )
        service.store.write_record(invalid_status_baseline)
        with pytest.raises(ApplicationServiceError, match="evidence_status_invalid"):
            service.record_result(
                replace(
                    valid,
                    result_id=f"result-binding-{invalid_status.value}",
                    baseline_comparisons=(
                        replace(
                            comparison,
                            baseline=record_reference(invalid_status_baseline),
                        ),
                    ),
                )
            )

    failed_integrity_baseline = replace(
        baseline,
        result_id="result-binding-failed-integrity-baseline",
        artifact_integrity_status=ArtifactIntegrityStatus.FAILED,
    )
    service.store.write_record(failed_integrity_baseline)
    with pytest.raises(ApplicationServiceError, match="artifact_integrity_invalid"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-failed-integrity",
                baseline_comparisons=(
                    replace(
                        comparison,
                        baseline=record_reference(failed_integrity_baseline),
                    ),
                ),
            )
        )

    mismatched_integrity_baseline = replace(
        baseline,
        result_id="result-binding-mismatched-integrity-baseline",
        artifact_integrity_evidence=_identity("mismatched-baseline-integrity"),
    )
    service.store.write_record(mismatched_integrity_baseline)
    with pytest.raises(
        ApplicationServiceError,
        match="artifact_integrity_evidence_mismatch",
    ):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-mismatched-integrity",
                baseline_comparisons=(
                    replace(
                        comparison,
                        baseline=record_reference(mismatched_integrity_baseline),
                    ),
                ),
            )
        )

    with pytest.raises(ApplicationServiceError, match="record_reference_not_found"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-missing-baseline",
                baseline_comparisons=(
                    replace(
                        comparison,
                        baseline=RecordReference(
                            "evaluation_result",
                            "result-binding-missing",
                            _identity("result-binding-missing"),
                        ),
                    ),
                ),
            )
        )

    failed_policy_baseline = service.record_result(
        replace(
            baseline,
            result_id="result-binding-objective-policy-failure",
            evidence_status=EvidenceStatus.FAILED,
        )
    )
    comparable_to_failed = service.record_result(
        replace(
            valid,
            result_id="result-binding-objective-policy-failure-comparison",
            baseline_comparisons=(
                replace(
                    comparison,
                    baseline=record_reference(failed_policy_baseline),
                ),
            ),
        )
    )
    assert (
        comparable_to_failed.baseline_comparisons[0].outcome is BaselineOutcome.BETTER
    )

    assert service.record_result(valid) == valid

    missing_artifact_baseline = replace(
        baseline,
        result_id="result-binding-missing-artifact-baseline",
        candidate=RecordReference(
            "artifact",
            "artifact-binding-missing",
            _identity("artifact-binding-missing"),
        ),
    )
    service.store.write_record(missing_artifact_baseline)
    with pytest.raises(ApplicationServiceError, match="record_reference_not_found"):
        service.record_result(
            replace(
                valid,
                result_id="result-binding-missing-artifact",
                baseline_comparisons=(
                    replace(
                        comparison,
                        baseline=record_reference(missing_artifact_baseline),
                    ),
                ),
            )
        )
