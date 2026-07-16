from dataclasses import replace
import hashlib
from pathlib import Path
import socket

from temper_ml.app_services.evaluations import EvaluationService
from temper_ml.cli import main
from temper_ml.domain.artifacts import Artifact
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
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import record_reference
from temper_ml.domain.runs import EvaluationMode
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _suite(suite_id: str, kind: CaseSuiteKind) -> EvaluationSuite:
    return EvaluationSuite(
        suite_id,
        kind,
        SuiteEvidenceState.UNSEALED,
        (
            EvaluationCase(f"case-{kind.value}-one", _identity(f"{kind.value}:one")),
            EvaluationCase(f"case-{kind.value}-two", _identity(f"{kind.value}:two")),
        ),
        (
            EvaluatorSpec(
                "task-accuracy",
                EvaluatorKind.TASK_METRIC,
                "accuracy",
                MetricDirection.MAXIMIZE,
            ),
            EvaluatorSpec(
                "format-validity",
                EvaluatorKind.FORMAT_CHECK,
                "format_validity",
                MetricDirection.MAXIMIZE,
            ),
        ),
    )


def _quality_result(
    result_id: str,
    artifact: Artifact,
    suite: EvaluationSuite,
    status: EvidenceStatus,
    accuracy: int,
) -> EvaluationResult:
    return EvaluationResult(
        result_id,
        record_reference(artifact),
        EvaluationMode.FULL_SUITE,
        ArtifactIntegrityStatus.PASSED,
        artifact.integrity_evidence,
        status,
        metrics=(
            MetricObservation(
                "accuracy",
                EvaluatorKind.TASK_METRIC,
                accuracy,
                MetricDirection.MAXIMIZE,
            ),
            MetricObservation(
                "format_validity",
                EvaluatorKind.FORMAT_CHECK,
                1,
                MetricDirection.MAXIMIZE,
            ),
        ),
        suite=record_reference(suite),
        suite_state=suite.state,
    )


def test_slice_six_offline_evidence_graph_is_deterministic_and_public_safe(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    def network_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Slice 6 fixture attempted network access")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    assert main(["fixture-workflow", str(tmp_path)]) == 0
    capsys.readouterr()
    store = TypedEvidenceStore(tmp_path)
    source = next(
        stored.record
        for stored in store.iter_records()
        if isinstance(stored.record, Artifact)
    )
    artifacts = tuple(
        replace(source, artifact_id=f"artifact-slice-six-{index}")
        for index in range(1, 6)
    )
    for artifact in artifacts:
        store.write_record(artifact)

    service = EvaluationService(tmp_path)
    development = service.register_suite(
        _suite("suite-development", CaseSuiteKind.DEVELOPMENT)
    )
    regression = service.register_suite(
        _suite("suite-regression", CaseSuiteKind.REGRESSION)
    )
    confirmation = service.seal_suite(
        service.register_suite(_suite("suite-confirmation", CaseSuiteKind.CONFIRMATION))
    )
    assert development.state is SuiteEvidenceState.UNSEALED
    assert regression.state is SuiteEvidenceState.UNSEALED
    assert confirmation.state is SuiteEvidenceState.SEALED

    solo = service.record_solo_review(
        Review(
            "review-solo-slice-six",
            ReviewMode.SOLO,
            ReviewStage.RECORDED,
            (
                ReviewEntry(
                    "prompt-solo-slice-six",
                    {"text": "Synthetic solo-review prompt"},
                    {"temperature": 0, "maximum_tokens": 32},
                    (
                        ReviewOutput(
                            "candidate-001",
                            {"text": "Synthetic solo-review output"},
                        ),
                    ),
                    "The recorded synthetic output followed the requested structure.",
                    (ReviewRating("candidate-001", "task_fit", 1),),
                ),
            ),
            "I reviewed the recorded prompt, settings, and output.",
            (ReviewCandidate("candidate-001", record_reference(artifacts[2])),),
            False,
        )
    )
    passed = service.record_result(
        _quality_result(
            "result-slice-six-passed",
            artifacts[0],
            confirmation,
            EvidenceStatus.PASSED,
            9,
        )
    )
    failed = service.record_result(
        _quality_result(
            "result-slice-six-failed",
            artifacts[1],
            regression,
            EvidenceStatus.FAILED,
            6,
        )
    )
    subjective = service.record_result(
        EvaluationResult(
            "result-slice-six-subjective",
            record_reference(artifacts[2]),
            EvaluationMode.LIGHT_EVALUATION,
            ArtifactIntegrityStatus.PASSED,
            artifacts[2].integrity_evidence,
            EvidenceStatus.SUBJECTIVE_ONLY,
            review=record_reference(solo),
        )
    )
    unevaluated = service.record_result(
        EvaluationResult(
            "result-slice-six-unevaluated",
            record_reference(artifacts[3]),
            EvaluationMode.NO_QUALITY_EVALUATION,
            ArtifactIntegrityStatus.PASSED,
            artifacts[3].integrity_evidence,
            EvidenceStatus.UNEVALUATED,
        )
    )
    policy = service.register_policy(
        RecommendationPolicy(
            "policy-slice-six",
            (
                HardQualifier(
                    "format_validity",
                    ComparisonOperator.GREATER_THAN_OR_EQUAL,
                    1,
                ),
            ),
            ("accuracy",),
            (OptimizationObjective("accuracy", MetricDirection.MAXIMIZE),),
            (),
            (
                ConfidenceRule(
                    ConfidenceLabel.HIGH,
                    (EvidenceStatus.PASSED,),
                    (SuiteEvidenceState.SEALED,),
                    2,
                ),
                ConfidenceRule(
                    ConfidenceLabel.LOW,
                    (EvidenceStatus.SUBJECTIVE_ONLY,),
                ),
            ),
        )
    )
    first_recommendation = service.recommend(
        "recommendation-slice-six",
        policy,
        (subjective, failed, passed),
    )
    replayed_recommendation = service.recommend(
        "recommendation-slice-six",
        policy,
        (passed, failed, subjective),
    )
    override = service.record_decision(
        UserDecision(
            "decision-slice-six-override",
            record_reference(first_recommendation),
            failed.candidate,
            UserDecisionStatus.DEPLOYMENT_OVERRIDE,
            EvidenceStatus.FAILED,
        )
    )

    assert replayed_recommendation.identity == first_recommendation.identity
    assert first_recommendation.selected_candidate == passed.candidate
    assert first_recommendation.confidence is ConfidenceLabel.HIGH
    disclosures = {
        item.evidence_status: item.suite_state
        for item in first_recommendation.assessments
    }
    assert disclosures == {
        EvidenceStatus.PASSED: SuiteEvidenceState.SEALED,
        EvidenceStatus.FAILED: SuiteEvidenceState.UNSEALED,
        EvidenceStatus.SUBJECTIVE_ONLY: None,
    }
    assert override.override_reason is None
    assert override.evidence_status_at_decision is EvidenceStatus.FAILED
    assert unevaluated.evidence_status is EvidenceStatus.UNEVALUATED

    inspected = service.inspect_confirmation_suite(confirmation)
    contaminated = service.record_result(
        _quality_result(
            "result-slice-six-contaminated",
            artifacts[4],
            inspected,
            EvidenceStatus.CONTAMINATED,
            9,
        )
    )
    assert contaminated.suite_state is SuiteEvidenceState.UNSEALED

    verification = service.store.verify()
    assert verification.record_counts["evaluation_suite"] == 5
    assert verification.record_counts["evaluation_result"] == 5
    assert verification.record_counts["recommendation_policy"] == 1
    assert verification.record_counts["recommendation"] == 1
    assert verification.record_counts["user_decision"] == 1
    assert verification.record_counts["review"] == 1
    event_types = {
        event.event_type
        for stream in service.store.iter_streams()
        for event in stream.events
    }
    assert {
        "evaluation_suite_registered",
        "evaluation_suite_state_changed",
        "solo_review_recorded",
        "evaluation_result_recorded",
        "recommendation_policy_recorded",
        "recommendation_recorded",
        "user_decision_recorded",
    } <= event_types

    public = dumps_canonical_json(service.store.public_dump().value)
    assert b"Synthetic solo-review prompt" not in public
    assert b"Synthetic solo-review output" not in public
    assert str(tmp_path).encode() not in public
    assert b'"fields":{}' in public
