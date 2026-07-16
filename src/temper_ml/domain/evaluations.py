"""Immutable evaluation, recommendation, decision, and review evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Self

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    freeze_json_object,
    identity_fields,
    parse_identity,
    require_identifier,
    require_non_negative_int,
    require_positive_int,
    require_string_tuple,
    require_text,
    thaw_json,
)
from temper_ml.domain.runs import EvaluationMode

Numeric = int | Decimal
CASE_MEMBERSHIP_PROJECTION = HashProjection("evaluation.case_membership", "v1")


class CaseSuiteKind(str, Enum):
    DEVELOPMENT = "development"
    REGRESSION = "regression"
    CONFIRMATION = "confirmation"


class SuiteEvidenceState(str, Enum):
    SEALED = "sealed"
    UNSEALED = "unsealed"
    MODIFIED = "modified"
    CONTAMINATED = "contaminated"
    RETIRED = "retired"


class EvaluatorKind(str, Enum):
    DETERMINISTIC_CHECK = "deterministic_check"
    HELD_OUT_LOSS = "held_out_loss"
    TASK_METRIC = "task_metric"
    FORMAT_CHECK = "format_check"
    MODEL_JUDGE = "model_judge"


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ArtifactIntegrityStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class EvidenceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    SUBJECTIVE_ONLY = "subjective_only"
    UNEVALUATED = "unevaluated"
    CONTAMINATED = "contaminated"


class BaselineOutcome(str, Enum):
    BETTER = "better"
    EQUIVALENT = "equivalent"
    WORSE = "worse"
    INCONCLUSIVE = "inconclusive"


class ComparisonOperator(str, Enum):
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUAL = "equal"


class ConfidenceLabel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class UserDecisionStatus(str, Enum):
    SELECTED = "selected"
    REJECTED = "rejected"
    PINNED = "pinned"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DEPLOYMENT_OVERRIDE = "deployment_override"


class ReviewMode(str, Enum):
    SOLO = "solo"
    BLIND = "blind"


class ReviewStage(str, Enum):
    RECORDED = "recorded"
    BLIND_PREPARED = "blind_prepared"
    BLIND_SEALED = "blind_sealed"
    BLIND_REVEALED = "blind_revealed"


def _require_numeric(field: str, value: Numeric) -> Numeric:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise RecordValidationError(f"{field} must be a canonical number")
    if isinstance(value, Decimal) and not value.is_finite():
        raise RecordValidationError(f"{field} must be finite")
    return value


def _require_non_negative_numeric(field: str, value: Numeric) -> Numeric:
    _require_numeric(field, value)
    if value < 0:
        raise RecordValidationError(f"{field} must be non-negative")
    return value


def _as_decimal(value: Numeric) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(value)


def _require_reference(
    field: str, value: RecordReference, record_type: str
) -> RecordReference:
    if not isinstance(value, RecordReference) or value.record_type != record_type:
        raise RecordValidationError(f"{field} must reference {record_type}")
    return value


def _reference_key(reference: RecordReference) -> tuple[str, str, str]:
    return (reference.record_type, reference.logical_id, reference.identity.value)


def _sorted_unique_text(field: str, values: tuple[str, ...]) -> tuple[str, ...]:
    return require_string_tuple(
        field,
        values,
        non_empty=False,
        sorted_values=True,
    )


@dataclass(frozen=True)
class EvaluationCase:
    """One public-safe case identifier bound to exact immutable case content."""

    case_id: str
    content_identity: ContentIdentity

    def __post_init__(self) -> None:
        require_identifier("case_id", self.case_id)
        if not isinstance(self.content_identity, ContentIdentity):
            raise RecordValidationError("case content_identity must be an identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "content_identity": identity_fields(self.content_identity),
        }


@dataclass(frozen=True)
class EvaluatorSpec:
    """A deterministic v1 evaluator declaration."""

    evaluator_id: str
    kind: EvaluatorKind
    metric_name: str
    direction: MetricDirection

    def __post_init__(self) -> None:
        require_identifier("evaluator_id", self.evaluator_id)
        require_identifier("metric_name", self.metric_name)
        if not isinstance(self.kind, EvaluatorKind):
            raise RecordValidationError("evaluator kind is invalid")
        if self.kind is EvaluatorKind.MODEL_JUDGE:
            raise RecordValidationError("model-judge evaluators are not allowed in v1")
        if not isinstance(self.direction, MetricDirection):
            raise RecordValidationError("metric direction is invalid")
        if self.metric_name in {"aggregate_quality_score", "universal_quality_score"}:
            raise RecordValidationError("universal aggregate scores are not allowed")

    def to_dict(self) -> dict[str, str]:
        return {
            "evaluator_id": self.evaluator_id,
            "kind": self.kind.value,
            "metric_name": self.metric_name,
            "direction": self.direction.value,
        }


@dataclass(frozen=True)
class EvaluationSuite(TypedRecord):
    """One immutable suite revision with deterministic case membership."""

    RECORD_TYPE: ClassVar[str] = "evaluation_suite"

    suite_id: str
    kind: CaseSuiteKind
    state: SuiteEvidenceState
    cases: tuple[EvaluationCase, ...]
    evaluators: tuple[EvaluatorSpec, ...]
    prior_suite: RecordReference | None = None

    def __post_init__(self) -> None:
        require_identifier("suite_id", self.suite_id)
        if not isinstance(self.kind, CaseSuiteKind):
            raise RecordValidationError("suite kind is invalid")
        if not isinstance(self.state, SuiteEvidenceState):
            raise RecordValidationError("suite evidence state is invalid")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise RecordValidationError("evaluation suite cases must be non-empty")
        if any(not isinstance(case, EvaluationCase) for case in self.cases):
            raise RecordValidationError("evaluation suite contains an invalid case")
        ordered_cases = tuple(sorted(self.cases, key=lambda case: case.case_id))
        case_ids = tuple(case.case_id for case in ordered_cases)
        if len(set(case_ids)) != len(case_ids):
            raise RecordValidationError("evaluation case ids must be unique")
        if len({case.content_identity for case in ordered_cases}) != len(ordered_cases):
            raise RecordValidationError("evaluation case content must be unique")
        object.__setattr__(self, "cases", ordered_cases)
        if not isinstance(self.evaluators, tuple) or not self.evaluators:
            raise RecordValidationError("evaluation suite evaluators must be non-empty")
        if any(
            not isinstance(evaluator, EvaluatorSpec) for evaluator in self.evaluators
        ):
            raise RecordValidationError(
                "evaluation suite contains an invalid evaluator"
            )
        ordered_evaluators = tuple(
            sorted(self.evaluators, key=lambda evaluator: evaluator.evaluator_id)
        )
        evaluator_ids = tuple(item.evaluator_id for item in ordered_evaluators)
        metric_names = tuple(item.metric_name for item in ordered_evaluators)
        if len(set(evaluator_ids)) != len(evaluator_ids):
            raise RecordValidationError("evaluator ids must be unique")
        if len(set(metric_names)) != len(metric_names):
            raise RecordValidationError("evaluator metric names must be unique")
        object.__setattr__(self, "evaluators", ordered_evaluators)
        if self.prior_suite is not None:
            _require_reference("prior_suite", self.prior_suite, self.RECORD_TYPE)
            if self.prior_suite.logical_id != self.suite_id:
                raise RecordValidationError("prior_suite must retain the suite id")
        elif self.state is not SuiteEvidenceState.UNSEALED:
            raise RecordValidationError(
                "the initial evaluation suite revision must be unsealed"
            )

    @property
    def case_membership_identity(self) -> ContentIdentity:
        return content_identity(
            CASE_MEMBERSHIP_PROJECTION,
            {"cases": [case.to_dict() for case in self.cases]},
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "kind": self.kind.value,
            "state": self.state.value,
            "case_membership_identity": identity_fields(self.case_membership_identity),
            "cases": [case.to_dict() for case in self.cases],
            "evaluators": [evaluator.to_dict() for evaluator in self.evaluators],
            "prior_suite": (
                self.prior_suite.to_dict() if self.prior_suite is not None else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping) or set(payload) != {
            "suite_id",
            "kind",
            "state",
            "case_membership_identity",
            "cases",
            "evaluators",
            "prior_suite",
        }:
            raise RecordValidationError("evaluation_suite fields are invalid")
        ordinary = dict(payload)
        claimed = ordinary.pop("case_membership_identity")
        suite = super().from_payload(ordinary)
        if (
            not isinstance(claimed, Mapping)
            or parse_identity(
                claimed, field="evaluation_suite.case_membership_identity"
            )
            != suite.case_membership_identity
        ):
            raise RecordValidationError("case membership identity mismatch")
        return suite


@dataclass(frozen=True)
class MetricObservation:
    """One evaluator-specific metric without aggregation into a hidden score."""

    metric_name: str
    evaluator_kind: EvaluatorKind
    value: Numeric
    direction: MetricDirection

    def __post_init__(self) -> None:
        require_identifier("metric_name", self.metric_name)
        if not isinstance(self.evaluator_kind, EvaluatorKind):
            raise RecordValidationError("metric evaluator kind is invalid")
        if self.evaluator_kind is EvaluatorKind.MODEL_JUDGE:
            raise RecordValidationError("model-judge evidence is not allowed in v1")
        if not isinstance(self.direction, MetricDirection):
            raise RecordValidationError("metric direction is invalid")
        _require_numeric("metric value", self.value)
        if self.metric_name in {"aggregate_quality_score", "universal_quality_score"}:
            raise RecordValidationError("universal aggregate scores are not allowed")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "evaluator_kind": self.evaluator_kind.value,
            "value": self.value,
            "direction": self.direction.value,
        }


@dataclass(frozen=True)
class BaselineComparison:
    """Candidate-versus-baseline evidence retained independently by metric."""

    comparison_id: str
    metric_name: str
    baseline: RecordReference
    candidate_value: Numeric
    baseline_value: Numeric
    outcome: BaselineOutcome

    def __post_init__(self) -> None:
        require_identifier("comparison_id", self.comparison_id)
        require_identifier("metric_name", self.metric_name)
        _require_reference("baseline", self.baseline, "evaluation_result")
        _require_numeric("candidate_value", self.candidate_value)
        _require_numeric("baseline_value", self.baseline_value)
        if not isinstance(self.outcome, BaselineOutcome):
            raise RecordValidationError("baseline outcome is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_id": self.comparison_id,
            "metric_name": self.metric_name,
            "baseline": self.baseline.to_dict(),
            "candidate_value": self.candidate_value,
            "baseline_value": self.baseline_value,
            "outcome": self.outcome.value,
        }


@dataclass(frozen=True)
class EvaluationResult(TypedRecord):
    """Honest quality evidence kept separate from artifact-integrity evidence."""

    RECORD_TYPE: ClassVar[str] = "evaluation_result"

    result_id: str
    candidate: RecordReference
    evaluation_mode: EvaluationMode
    artifact_integrity_status: ArtifactIntegrityStatus
    artifact_integrity_evidence: ContentIdentity
    evidence_status: EvidenceStatus
    metrics: tuple[MetricObservation, ...] = ()
    baseline_comparisons: tuple[BaselineComparison, ...] = ()
    suite: RecordReference | None = None
    suite_state: SuiteEvidenceState | None = None
    review: RecordReference | None = None
    conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier("result_id", self.result_id)
        _require_reference("candidate", self.candidate, "artifact")
        if not isinstance(self.evaluation_mode, EvaluationMode):
            raise RecordValidationError("evaluation mode is invalid")
        if not isinstance(self.artifact_integrity_status, ArtifactIntegrityStatus):
            raise RecordValidationError("artifact integrity status is invalid")
        if not isinstance(self.artifact_integrity_evidence, ContentIdentity):
            raise RecordValidationError(
                "artifact_integrity_evidence must be a content identity"
            )
        if not isinstance(self.evidence_status, EvidenceStatus):
            raise RecordValidationError("evidence status is invalid")
        if not isinstance(self.metrics, tuple) or any(
            not isinstance(metric, MetricObservation) for metric in self.metrics
        ):
            raise RecordValidationError("metrics must be metric observations")
        ordered_metrics = tuple(sorted(self.metrics, key=lambda item: item.metric_name))
        if len({item.metric_name for item in ordered_metrics}) != len(ordered_metrics):
            raise RecordValidationError("metric names must be unique")
        object.__setattr__(self, "metrics", ordered_metrics)
        if not isinstance(self.baseline_comparisons, tuple) or any(
            not isinstance(comparison, BaselineComparison)
            for comparison in self.baseline_comparisons
        ):
            raise RecordValidationError(
                "baseline_comparisons must be baseline comparison evidence"
            )
        ordered_comparisons = tuple(
            sorted(self.baseline_comparisons, key=lambda item: item.comparison_id)
        )
        if len({item.comparison_id for item in ordered_comparisons}) != len(
            ordered_comparisons
        ):
            raise RecordValidationError("baseline comparison ids must be unique")
        object.__setattr__(self, "baseline_comparisons", ordered_comparisons)
        if (self.suite is None) != (self.suite_state is None):
            raise RecordValidationError(
                "suite and suite_state must be supplied together"
            )
        if self.suite is not None:
            _require_reference("suite", self.suite, "evaluation_suite")
        if self.suite_state is not None and not isinstance(
            self.suite_state, SuiteEvidenceState
        ):
            raise RecordValidationError("suite_state is invalid")
        suite_backed_modes = (
            EvaluationMode.FULL_SUITE,
            EvaluationMode.EXPERIMENT_LOOP,
        )
        if self.evaluation_mode in suite_backed_modes and self.suite is None:
            raise RecordValidationError(
                "suite-backed evaluation modes require suite evidence"
            )
        if self.evaluation_mode not in suite_backed_modes and self.suite is not None:
            raise RecordValidationError(
                "non-suite evaluation modes must not reference suite evidence"
            )
        if self.baseline_comparisons and self.suite is None:
            raise RecordValidationError(
                "baseline comparisons require shared suite evidence"
            )
        if self.review is not None:
            _require_reference("review", self.review, "review")
        object.__setattr__(
            self,
            "conflicts",
            _sorted_unique_text("conflicts", self.conflicts),
        )
        if self.evaluation_mode is EvaluationMode.NO_QUALITY_EVALUATION:
            if (
                self.evidence_status is not EvidenceStatus.UNEVALUATED
                or self.metrics
                or self.baseline_comparisons
                or self.suite is not None
                or self.review is not None
            ):
                raise RecordValidationError(
                    "no-quality mode must retain only separate integrity evidence"
                )
        if (
            self.evidence_status is EvidenceStatus.SUBJECTIVE_ONLY
            and self.review is None
        ):
            raise RecordValidationError("subjective-only evidence requires a review")
        if self.evidence_status in (
            EvidenceStatus.PASSED,
            EvidenceStatus.FAILED,
        ) and not (self.metrics or self.baseline_comparisons):
            raise RecordValidationError(
                "passed or failed evidence requires measurements"
            )
        if (
            self.suite_state is SuiteEvidenceState.CONTAMINATED
            and self.evidence_status is not EvidenceStatus.CONTAMINATED
        ):
            raise RecordValidationError(
                "a contaminated suite must produce contaminated evidence"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "candidate": self.candidate.to_dict(),
            "evaluation_mode": self.evaluation_mode.value,
            "artifact_integrity_status": self.artifact_integrity_status.value,
            "artifact_integrity_evidence": identity_fields(
                self.artifact_integrity_evidence
            ),
            "evidence_status": self.evidence_status.value,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "baseline_comparisons": [
                comparison.to_dict() for comparison in self.baseline_comparisons
            ],
            "suite": self.suite.to_dict() if self.suite is not None else None,
            "suite_state": (
                self.suite_state.value if self.suite_state is not None else None
            ),
            "review": self.review.to_dict() if self.review is not None else None,
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class HardQualifier:
    metric_name: str
    operator: ComparisonOperator
    threshold: Numeric
    tolerance: Numeric = 0

    def __post_init__(self) -> None:
        require_identifier("metric_name", self.metric_name)
        if not isinstance(self.operator, ComparisonOperator):
            raise RecordValidationError("qualifier operator is invalid")
        _require_numeric("qualifier threshold", self.threshold)
        _require_non_negative_numeric("qualifier tolerance", self.tolerance)

    def accepts(self, value: Numeric) -> bool:
        _require_numeric("qualifier value", value)
        comparable = _as_decimal(value)
        threshold = _as_decimal(self.threshold)
        tolerance = _as_decimal(self.tolerance)
        if self.operator is ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return comparable >= threshold - tolerance
        if self.operator is ComparisonOperator.LESS_THAN_OR_EQUAL:
            return comparable <= threshold + tolerance
        if self.operator is ComparisonOperator.GREATER_THAN:
            return comparable > threshold + tolerance
        if self.operator is ComparisonOperator.LESS_THAN:
            return comparable < threshold - tolerance
        return abs(comparable - threshold) <= tolerance

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class OptimizationObjective:
    metric_name: str
    direction: MetricDirection
    tie_tolerance: Numeric = 0

    def __post_init__(self) -> None:
        require_identifier("metric_name", self.metric_name)
        if not isinstance(self.direction, MetricDirection):
            raise RecordValidationError("objective direction is invalid")
        _require_non_negative_numeric("objective tie_tolerance", self.tie_tolerance)

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "tie_tolerance": self.tie_tolerance,
        }


@dataclass(frozen=True)
class ConfidenceRule:
    label: ConfidenceLabel
    evidence_statuses: tuple[EvidenceStatus, ...]
    suite_states: tuple[SuiteEvidenceState, ...] = ()
    minimum_metric_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.label, ConfidenceLabel):
            raise RecordValidationError("confidence label is invalid")
        if not isinstance(self.evidence_statuses, tuple) or not self.evidence_statuses:
            raise RecordValidationError(
                "confidence evidence statuses must be non-empty"
            )
        if any(
            not isinstance(status, EvidenceStatus) for status in self.evidence_statuses
        ):
            raise RecordValidationError("confidence evidence status is invalid")
        if len(set(self.evidence_statuses)) != len(self.evidence_statuses):
            raise RecordValidationError("confidence evidence statuses must be unique")
        object.__setattr__(
            self,
            "evidence_statuses",
            tuple(sorted(self.evidence_statuses, key=lambda status: status.value)),
        )
        if not isinstance(self.suite_states, tuple) or any(
            not isinstance(state, SuiteEvidenceState) for state in self.suite_states
        ):
            raise RecordValidationError("confidence suite states are invalid")
        if len(set(self.suite_states)) != len(self.suite_states):
            raise RecordValidationError("confidence suite states must be unique")
        object.__setattr__(
            self,
            "suite_states",
            tuple(sorted(self.suite_states, key=lambda state: state.value)),
        )
        require_non_negative_int("minimum_metric_count", self.minimum_metric_count)

    def matches(self, result: EvaluationResult) -> bool:
        return (
            result.evidence_status in self.evidence_statuses
            and len(result.metrics) >= self.minimum_metric_count
            and (
                not self.suite_states
                or (
                    result.suite_state is not None
                    and result.suite_state in self.suite_states
                )
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label.value,
            "evidence_statuses": [status.value for status in self.evidence_statuses],
            "suite_states": [state.value for state in self.suite_states],
            "minimum_metric_count": self.minimum_metric_count,
        }


@dataclass(frozen=True)
class RecommendationPolicy(TypedRecord):
    """Explicit qualifiers and ordered objectives for deterministic selection."""

    RECORD_TYPE: ClassVar[str] = "recommendation_policy"

    policy_id: str
    hard_qualifiers: tuple[HardQualifier, ...]
    advisory_metrics: tuple[str, ...]
    objectives: tuple[OptimizationObjective, ...]
    baseline_comparisons: tuple[str, ...]
    confidence_rules: tuple[ConfidenceRule, ...]

    def __post_init__(self) -> None:
        require_identifier("policy_id", self.policy_id)
        if not isinstance(self.hard_qualifiers, tuple) or any(
            not isinstance(item, HardQualifier) for item in self.hard_qualifiers
        ):
            raise RecordValidationError("hard qualifiers are invalid")
        qualifier_names = tuple(item.metric_name for item in self.hard_qualifiers)
        if len(set(qualifier_names)) != len(qualifier_names):
            raise RecordValidationError("hard qualifier metrics must be unique")
        object.__setattr__(
            self,
            "hard_qualifiers",
            tuple(sorted(self.hard_qualifiers, key=lambda item: item.metric_name)),
        )
        object.__setattr__(
            self,
            "advisory_metrics",
            _sorted_unique_text("advisory_metrics", self.advisory_metrics),
        )
        if (
            not isinstance(self.objectives, tuple)
            or not self.objectives
            or any(
                not isinstance(item, OptimizationObjective) for item in self.objectives
            )
        ):
            raise RecordValidationError("ordered objectives must be non-empty")
        objective_names = tuple(item.metric_name for item in self.objectives)
        if len(set(objective_names)) != len(objective_names):
            raise RecordValidationError("objective metrics must be unique")
        object.__setattr__(
            self,
            "baseline_comparisons",
            _sorted_unique_text("baseline_comparisons", self.baseline_comparisons),
        )
        if not isinstance(self.confidence_rules, tuple) or not self.confidence_rules:
            raise RecordValidationError("confidence rules must be non-empty")
        if any(not isinstance(item, ConfidenceRule) for item in self.confidence_rules):
            raise RecordValidationError("confidence rules are invalid")
        if len({item.label for item in self.confidence_rules}) != len(
            self.confidence_rules
        ):
            raise RecordValidationError("confidence rule labels must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "hard_qualifiers": [item.to_dict() for item in self.hard_qualifiers],
            "advisory_metrics": list(self.advisory_metrics),
            "objectives": [item.to_dict() for item in self.objectives],
            "baseline_comparisons": list(self.baseline_comparisons),
            "confidence_rules": [item.to_dict() for item in self.confidence_rules],
        }


@dataclass(frozen=True)
class ObjectiveValue:
    metric_name: str
    value: Numeric

    def __post_init__(self) -> None:
        require_identifier("metric_name", self.metric_name)
        _require_numeric("objective value", self.value)

    def to_dict(self) -> dict[str, object]:
        return {"metric_name": self.metric_name, "value": self.value}


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: RecordReference
    evaluation_result: RecordReference
    evidence_status: EvidenceStatus
    suite_state: SuiteEvidenceState | None
    qualified: bool
    rank: int | None
    qualifier_failures: tuple[str, ...]
    objective_values: tuple[ObjectiveValue, ...]

    def __post_init__(self) -> None:
        _require_reference("candidate", self.candidate, "artifact")
        _require_reference(
            "evaluation_result", self.evaluation_result, "evaluation_result"
        )
        if not isinstance(self.evidence_status, EvidenceStatus):
            raise RecordValidationError("assessment evidence status is invalid")
        if self.suite_state is not None and not isinstance(
            self.suite_state, SuiteEvidenceState
        ):
            raise RecordValidationError("assessment suite state is invalid")
        if not isinstance(self.qualified, bool):
            raise RecordValidationError("assessment qualified must be boolean")
        if self.rank is not None:
            require_positive_int("assessment rank", self.rank)
        if self.qualified != (self.rank is not None):
            raise RecordValidationError("only qualified candidates may have a rank")
        object.__setattr__(
            self,
            "qualifier_failures",
            _sorted_unique_text("qualifier_failures", self.qualifier_failures),
        )
        if self.qualified and self.qualifier_failures:
            raise RecordValidationError("qualified candidate cannot have failures")
        if not isinstance(self.objective_values, tuple) or any(
            not isinstance(value, ObjectiveValue) for value in self.objective_values
        ):
            raise RecordValidationError("assessment objective values are invalid")
        if len({item.metric_name for item in self.objective_values}) != len(
            self.objective_values
        ):
            raise RecordValidationError("assessment objective values must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "evaluation_result": self.evaluation_result.to_dict(),
            "evidence_status": self.evidence_status.value,
            "suite_state": self.suite_state.value if self.suite_state else None,
            "qualified": self.qualified,
            "rank": self.rank,
            "qualifier_failures": list(self.qualifier_failures),
            "objective_values": [value.to_dict() for value in self.objective_values],
        }


@dataclass(frozen=True)
class Recommendation(TypedRecord):
    """Policy-derived ranking and evidence disclosure, never a user decision."""

    RECORD_TYPE: ClassVar[str] = "recommendation"

    recommendation_id: str
    policy: RecordReference
    assessments: tuple[CandidateAssessment, ...]
    selected_candidate: RecordReference | None
    pareto_alternatives: tuple[RecordReference, ...]
    conflicts: tuple[str, ...]
    confidence: ConfidenceLabel

    def __post_init__(self) -> None:
        require_identifier("recommendation_id", self.recommendation_id)
        _require_reference("policy", self.policy, "recommendation_policy")
        if (
            not isinstance(self.assessments, tuple)
            or not self.assessments
            or any(
                not isinstance(item, CandidateAssessment) for item in self.assessments
            )
        ):
            raise RecordValidationError("recommendation assessments are invalid")
        ordered_assessments = tuple(
            sorted(self.assessments, key=lambda item: _reference_key(item.candidate))
        )
        object.__setattr__(self, "assessments", ordered_assessments)
        candidate_keys = tuple(
            _reference_key(item.candidate) for item in ordered_assessments
        )
        if len(set(candidate_keys)) != len(candidate_keys):
            raise RecordValidationError("recommendation candidates must be unique")
        ranked = sorted(
            (item for item in self.assessments if item.qualified),
            key=lambda item: item.rank if item.rank is not None else 0,
        )
        if tuple(item.rank for item in ranked) != tuple(range(1, len(ranked) + 1)):
            raise RecordValidationError("qualified ranks must be contiguous")
        expected_selected = ranked[0].candidate if ranked else None
        if self.selected_candidate != expected_selected:
            raise RecordValidationError("selected candidate must be rank one")
        alternatives = tuple(sorted(self.pareto_alternatives, key=_reference_key))
        if any(
            not isinstance(reference, RecordReference)
            or reference.record_type != "artifact"
            for reference in alternatives
        ):
            raise RecordValidationError("pareto alternatives must reference artifacts")
        if len(set(_reference_key(item) for item in alternatives)) != len(alternatives):
            raise RecordValidationError("pareto alternatives must be unique")
        qualified_keys = {_reference_key(item.candidate) for item in ranked}
        if any(_reference_key(item) not in qualified_keys for item in alternatives):
            raise RecordValidationError("pareto alternatives must be qualified")
        if self.selected_candidate in alternatives:
            raise RecordValidationError("selected candidate is not an alternative")
        object.__setattr__(self, "pareto_alternatives", alternatives)
        object.__setattr__(
            self,
            "conflicts",
            _sorted_unique_text("conflicts", self.conflicts),
        )
        if not isinstance(self.confidence, ConfidenceLabel):
            raise RecordValidationError("recommendation confidence is invalid")
        if (
            self.selected_candidate is None
            and self.confidence is not ConfidenceLabel.NONE
        ):
            raise RecordValidationError("no selection must have no confidence")

    def to_payload(self) -> dict[str, object]:
        return {
            "recommendation_id": self.recommendation_id,
            "policy": self.policy.to_dict(),
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "pareto_alternatives": [
                alternative.to_dict() for alternative in self.pareto_alternatives
            ],
            "conflicts": list(self.conflicts),
            "confidence": self.confidence.value,
        }


@dataclass(frozen=True)
class UserDecision(TypedRecord):
    """A user action recorded separately from immutable recommendation evidence."""

    RECORD_TYPE: ClassVar[str] = "user_decision"

    decision_id: str
    recommendation: RecordReference
    candidate: RecordReference
    status: UserDecisionStatus
    evidence_status_at_decision: EvidenceStatus
    override_reason: str | None = None

    def __post_init__(self) -> None:
        require_identifier("decision_id", self.decision_id)
        _require_reference("recommendation", self.recommendation, "recommendation")
        _require_reference("candidate", self.candidate, "artifact")
        if not isinstance(self.status, UserDecisionStatus):
            raise RecordValidationError("user decision status is invalid")
        if not isinstance(self.evidence_status_at_decision, EvidenceStatus):
            raise RecordValidationError("decision evidence status is invalid")
        if self.override_reason is not None:
            require_text("override_reason", self.override_reason)

    def to_payload(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "recommendation": self.recommendation.to_dict(),
            "candidate": self.candidate.to_dict(),
            "status": self.status.value,
            "evidence_status_at_decision": self.evidence_status_at_decision.value,
            "override_reason": self.override_reason,
        }


@dataclass(frozen=True)
class ReviewOutput:
    alias: str
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_identifier("review output alias", self.alias)
        object.__setattr__(
            self,
            "output",
            freeze_json_object(self.output, field="review output"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"alias": self.alias, "output": thaw_json(self.output)}


@dataclass(frozen=True)
class ReviewRating:
    alias: str
    criterion: str
    rating: Numeric

    def __post_init__(self) -> None:
        require_identifier("review rating alias", self.alias)
        require_identifier("review rating criterion", self.criterion)
        _require_numeric("review rating", self.rating)

    def to_dict(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "criterion": self.criterion,
            "rating": self.rating,
        }


@dataclass(frozen=True)
class ReviewEntry:
    prompt_id: str
    prompt: Mapping[str, Any]
    settings: Mapping[str, Any]
    outputs: tuple[ReviewOutput, ...]
    notes: str
    ratings: tuple[ReviewRating, ...]

    def __post_init__(self) -> None:
        require_identifier("review prompt_id", self.prompt_id)
        object.__setattr__(
            self,
            "prompt",
            freeze_json_object(self.prompt, field="review prompt"),
        )
        object.__setattr__(
            self,
            "settings",
            freeze_json_object(self.settings, field="review settings"),
        )
        if (
            not isinstance(self.outputs, tuple)
            or not self.outputs
            or any(not isinstance(output, ReviewOutput) for output in self.outputs)
        ):
            raise RecordValidationError("review outputs must be non-empty")
        ordered_outputs = tuple(sorted(self.outputs, key=lambda output: output.alias))
        aliases = tuple(output.alias for output in ordered_outputs)
        if len(set(aliases)) != len(aliases):
            raise RecordValidationError("review output aliases must be unique")
        object.__setattr__(self, "outputs", ordered_outputs)
        require_text("review notes", self.notes)
        if (
            not isinstance(self.ratings, tuple)
            or not self.ratings
            or any(not isinstance(rating, ReviewRating) for rating in self.ratings)
        ):
            raise RecordValidationError("review ratings must be non-empty")
        ordered_ratings = tuple(
            sorted(self.ratings, key=lambda rating: (rating.alias, rating.criterion))
        )
        rating_keys = tuple(
            (rating.alias, rating.criterion) for rating in ordered_ratings
        )
        if len(set(rating_keys)) != len(rating_keys):
            raise RecordValidationError("review ratings must be unique per criterion")
        if {rating.alias for rating in ordered_ratings} != set(aliases):
            raise RecordValidationError("every reviewed output requires a rating")
        object.__setattr__(self, "ratings", ordered_ratings)

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(output.alias for output in self.outputs)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": thaw_json(self.prompt),
            "settings": thaw_json(self.settings),
            "outputs": [output.to_dict() for output in self.outputs],
            "notes": self.notes,
            "ratings": [rating.to_dict() for rating in self.ratings],
        }


@dataclass(frozen=True)
class ReviewCandidate:
    alias: str
    candidate: RecordReference

    def __post_init__(self) -> None:
        require_identifier("review candidate alias", self.alias)
        _require_reference("review candidate", self.candidate, "artifact")

    def to_dict(self) -> dict[str, object]:
        return {"alias": self.alias, "candidate": self.candidate.to_dict()}


@dataclass(frozen=True)
class Review(TypedRecord):
    """Structured solo evidence or a sealed/revealed blind-review revision."""

    RECORD_TYPE: ClassVar[str] = "review"

    review_id: str
    mode: ReviewMode
    stage: ReviewStage
    entries: tuple[ReviewEntry, ...]
    reviewer_declaration: str | None
    candidate_mappings: tuple[ReviewCandidate, ...]
    leak_audit_passed: bool
    packet_identity: ContentIdentity | None = None
    prior_review: RecordReference | None = None
    hiding_nonce: str | None = None

    def __post_init__(self) -> None:
        require_identifier("review_id", self.review_id)
        if not isinstance(self.mode, ReviewMode):
            raise RecordValidationError("review mode is invalid")
        if not isinstance(self.stage, ReviewStage):
            raise RecordValidationError("review stage is invalid")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, ReviewEntry) for entry in self.entries
        ):
            raise RecordValidationError("review entries are invalid")
        ordered_entries = tuple(sorted(self.entries, key=lambda entry: entry.prompt_id))
        if len({entry.prompt_id for entry in ordered_entries}) != len(ordered_entries):
            raise RecordValidationError("review prompt ids must be unique")
        aliases = ordered_entries[0].aliases if ordered_entries else ()
        if ordered_entries and any(
            entry.aliases != aliases for entry in ordered_entries
        ):
            raise RecordValidationError("review entries must use the same aliases")
        object.__setattr__(self, "entries", ordered_entries)
        if self.reviewer_declaration is not None:
            require_text("reviewer_declaration", self.reviewer_declaration)
        if not isinstance(self.candidate_mappings, tuple) or any(
            not isinstance(mapping, ReviewCandidate)
            for mapping in self.candidate_mappings
        ):
            raise RecordValidationError("review candidate mappings are invalid")
        ordered_mappings = tuple(
            sorted(self.candidate_mappings, key=lambda mapping: mapping.alias)
        )
        if len({mapping.alias for mapping in ordered_mappings}) != len(
            ordered_mappings
        ):
            raise RecordValidationError("review candidate aliases must be unique")
        object.__setattr__(self, "candidate_mappings", ordered_mappings)
        if not isinstance(self.leak_audit_passed, bool):
            raise RecordValidationError("leak_audit_passed must be boolean")
        if self.packet_identity is not None and not isinstance(
            self.packet_identity, ContentIdentity
        ):
            raise RecordValidationError("packet_identity must be a content identity")
        if self.prior_review is not None:
            _require_reference("prior_review", self.prior_review, self.RECORD_TYPE)
            if self.prior_review.logical_id != self.review_id:
                raise RecordValidationError("prior review must retain the review id")
        if self.hiding_nonce is not None and (
            not isinstance(self.hiding_nonce, str)
            or len(self.hiding_nonce) != 64
            or any(
                character not in "0123456789abcdef" for character in self.hiding_nonce
            )
        ):
            raise RecordValidationError("hiding_nonce must be 256-bit lowercase hex")
        if self.mode is ReviewMode.SOLO:
            if (
                self.stage is not ReviewStage.RECORDED
                or not self.entries
                or self.reviewer_declaration is None
                or not self.candidate_mappings
                or tuple(mapping.alias for mapping in ordered_mappings) != aliases
                or self.leak_audit_passed
                or self.packet_identity is not None
                or self.prior_review is not None
                or self.hiding_nonce is not None
            ):
                raise RecordValidationError(
                    "solo reviews must be complete recorded evidence"
                )
        elif self.stage is ReviewStage.BLIND_PREPARED:
            if (
                self.entries
                or self.reviewer_declaration is not None
                or not self.candidate_mappings
                or not self.leak_audit_passed
                or self.packet_identity is None
                or self.prior_review is not None
                or self.hiding_nonce is None
            ):
                raise RecordValidationError(
                    "blind preparation requires private reveal material"
                )
        elif self.stage is ReviewStage.BLIND_SEALED:
            if (
                not self.entries
                or self.reviewer_declaration is None
                or self.candidate_mappings
                or not self.leak_audit_passed
                or self.packet_identity is None
                or self.prior_review is None
                or self.hiding_nonce is not None
            ):
                raise RecordValidationError(
                    "blind judgment must be sealed before reveal"
                )
        elif self.stage is ReviewStage.BLIND_REVEALED:
            if (
                not self.entries
                or self.reviewer_declaration is None
                or not self.candidate_mappings
                or tuple(mapping.alias for mapping in ordered_mappings) != aliases
                or not self.leak_audit_passed
                or self.packet_identity is None
                or self.prior_review is None
                or self.hiding_nonce is not None
            ):
                raise RecordValidationError(
                    "blind reveal requires sealed prior evidence"
                )
        else:
            raise RecordValidationError("blind review stage is invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "mode": self.mode.value,
            "stage": self.stage.value,
            "entries": [entry.to_dict() for entry in self.entries],
            "reviewer_declaration": self.reviewer_declaration,
            "candidate_mappings": [
                mapping.to_dict() for mapping in self.candidate_mappings
            ],
            "leak_audit_passed": self.leak_audit_passed,
            "packet_identity": (
                identity_fields(self.packet_identity)
                if self.packet_identity is not None
                else None
            ),
            "prior_review": (
                self.prior_review.to_dict() if self.prior_review is not None else None
            ),
            "hiding_nonce": self.hiding_nonce,
        }


def build_recommendation(
    recommendation_id: str,
    policy: RecommendationPolicy,
    results: tuple[EvaluationResult, ...],
) -> Recommendation:
    """Apply one policy with stable qualification, lexicographic, and Pareto rules."""

    require_identifier("recommendation_id", recommendation_id)
    if not isinstance(policy, RecommendationPolicy):
        raise RecordValidationError("recommendation policy is invalid")
    if (
        not isinstance(results, tuple)
        or not results
        or any(not isinstance(result, EvaluationResult) for result in results)
    ):
        raise RecordValidationError("recommendation results must be non-empty")
    ordered_results = tuple(
        sorted(results, key=lambda item: _reference_key(item.candidate))
    )
    if len({_reference_key(item.candidate) for item in ordered_results}) != len(
        ordered_results
    ):
        raise RecordValidationError("one result is allowed per candidate")

    failures: dict[tuple[str, str, str], tuple[str, ...]] = {}
    objective_values: dict[tuple[str, str, str], tuple[ObjectiveValue, ...]] = {}
    conflicts: set[str] = set()
    qualified_results: list[EvaluationResult] = []
    metric_semantics: dict[str, tuple[EvaluatorKind, MetricDirection]] = {}
    for result in ordered_results:
        key = _reference_key(result.candidate)
        observations = {metric.metric_name: metric for metric in result.metrics}
        metrics = {name: metric.value for name, metric in observations.items()}
        for metric in result.metrics:
            semantics = (metric.evaluator_kind, metric.direction)
            prior_semantics = metric_semantics.setdefault(metric.metric_name, semantics)
            if prior_semantics != semantics:
                raise RecordValidationError(
                    f"metric semantics conflict for {metric.metric_name}"
                )
        baseline_ids = {
            comparison.comparison_id for comparison in result.baseline_comparisons
        }
        candidate_failures: list[str] = []
        if result.artifact_integrity_status is not ArtifactIntegrityStatus.PASSED:
            candidate_failures.append("artifact_integrity:failed")
        if result.evidence_status not in (
            EvidenceStatus.PASSED,
            EvidenceStatus.SUBJECTIVE_ONLY,
        ):
            candidate_failures.append(f"evidence_status:{result.evidence_status.value}")
        for qualifier in policy.hard_qualifiers:
            value = metrics.get(qualifier.metric_name)
            if value is None:
                candidate_failures.append(f"missing_qualifier:{qualifier.metric_name}")
            elif not qualifier.accepts(value):
                candidate_failures.append(f"failed_qualifier:{qualifier.metric_name}")
        for comparison_id in policy.baseline_comparisons:
            if comparison_id not in baseline_ids:
                candidate_failures.append(f"missing_baseline:{comparison_id}")
        candidate_objectives: list[ObjectiveValue] = []
        for objective in policy.objectives:
            observation = observations.get(objective.metric_name)
            if observation is None:
                candidate_failures.append(f"missing_objective:{objective.metric_name}")
            elif observation.direction is not objective.direction:
                raise RecordValidationError(
                    f"objective direction conflicts with {objective.metric_name}"
                )
            else:
                candidate_objectives.append(
                    ObjectiveValue(objective.metric_name, observation.value)
                )
        for metric_name in policy.advisory_metrics:
            if metric_name not in metrics:
                conflicts.add(f"advisory_metric_missing:{metric_name}")
        for comparison in result.baseline_comparisons:
            if comparison.outcome is BaselineOutcome.WORSE:
                conflicts.add(f"baseline_worse:{comparison.comparison_id}")
            elif comparison.outcome is BaselineOutcome.INCONCLUSIVE:
                conflicts.add(f"baseline_inconclusive:{comparison.comparison_id}")
        conflicts.update(result.conflicts)
        failures[key] = tuple(sorted(set(candidate_failures)))
        objective_values[key] = tuple(candidate_objectives)
        if not candidate_failures:
            qualified_results.append(result)

    qualified = tuple(qualified_results)
    if qualified:
        cohort = (
            qualified[0].evaluation_mode,
            qualified[0].suite,
            qualified[0].suite_state,
        )
        if any(
            (result.evaluation_mode, result.suite, result.suite_state) != cohort
            for result in qualified[1:]
        ):
            raise RecordValidationError(
                "recommendation results must share one evaluation cohort"
            )
    cohort_keys = _global_tolerance_cohort_keys(qualified, policy)
    ranked = list(_rank_results(qualified, policy, cohort_keys=cohort_keys))
    rank_by_key = {
        _reference_key(result.candidate): rank for rank, result in enumerate(ranked, 1)
    }
    pareto = _pareto_results(qualified, policy, cohort_keys=cohort_keys)
    selected = ranked[0] if ranked else None
    selected_candidate = selected.candidate if selected is not None else None
    alternatives = tuple(
        result.candidate
        for result in pareto
        if selected_candidate is None or result.candidate != selected_candidate
    )
    if len(pareto) > 1:
        conflicts.add("qualified_objective_tradeoff")
    if selected is None:
        conflicts.add("no_qualified_candidate")

    assessments = tuple(
        CandidateAssessment(
            candidate=result.candidate,
            evaluation_result=RecordReference(
                "evaluation_result", result.result_id, result.identity
            ),
            evidence_status=result.evidence_status,
            suite_state=result.suite_state,
            qualified=_reference_key(result.candidate) in rank_by_key,
            rank=rank_by_key.get(_reference_key(result.candidate)),
            qualifier_failures=failures[_reference_key(result.candidate)],
            objective_values=objective_values[_reference_key(result.candidate)],
        )
        for result in ordered_results
    )
    confidence = ConfidenceLabel.NONE
    if selected is not None:
        for rule in policy.confidence_rules:
            if rule.matches(selected):
                confidence = rule.label
                break
    return Recommendation(
        recommendation_id=recommendation_id,
        policy=RecordReference(
            "recommendation_policy", policy.policy_id, policy.identity
        ),
        assessments=assessments,
        selected_candidate=selected_candidate,
        pareto_alternatives=alternatives,
        conflicts=tuple(conflicts),
        confidence=confidence,
    )


def _rank_results(
    results: tuple[EvaluationResult, ...],
    policy: RecommendationPolicy,
    *,
    cohort_keys: dict[tuple[str, str, str], tuple[int, ...]] | None = None,
) -> tuple[EvaluationResult, ...]:
    """Rank lexicographically with one global tolerance cohort per objective."""

    if cohort_keys is None:
        cohort_keys = _global_tolerance_cohort_keys(results, policy)
    return tuple(
        sorted(
            results,
            key=lambda item: (
                cohort_keys[_reference_key(item.candidate)],
                _reference_key(item.candidate),
            ),
        )
    )


def _global_tolerance_cohort_keys(
    results: tuple[EvaluationResult, ...],
    policy: RecommendationPolicy,
) -> dict[tuple[str, str, str], tuple[int, ...]]:
    """Assign deterministic best-anchored cohort indexes across all candidates."""

    cohort_keys: dict[tuple[str, str, str], list[int]] = {
        _reference_key(result.candidate): [] for result in results
    }
    for objective in policy.objectives:

        def objective_value(result: EvaluationResult) -> Decimal:
            metrics = {metric.metric_name: metric.value for metric in result.metrics}
            return _as_decimal(metrics[objective.metric_name])

        ordered = sorted(
            results,
            key=lambda item: (
                -objective_value(item)
                if objective.direction is MetricDirection.MAXIMIZE
                else objective_value(item),
                _reference_key(item.candidate),
            ),
        )
        tolerance = _as_decimal(objective.tie_tolerance)
        cursor = 0
        cohort_index = 0
        while cursor < len(ordered):
            anchor = objective_value(ordered[cursor])
            end = cursor + 1
            while (
                end < len(ordered)
                and abs(objective_value(ordered[end]) - anchor) <= tolerance
            ):
                end += 1
            for result in ordered[cursor:end]:
                cohort_keys[_reference_key(result.candidate)].append(cohort_index)
            cursor = end
            cohort_index += 1
    return {key: tuple(indexes) for key, indexes in cohort_keys.items()}


def _pareto_results(
    results: tuple[EvaluationResult, ...],
    policy: RecommendationPolicy,
    *,
    cohort_keys: dict[tuple[str, str, str], tuple[int, ...]] | None = None,
) -> tuple[EvaluationResult, ...]:
    if cohort_keys is None:
        cohort_keys = _global_tolerance_cohort_keys(results, policy)
    nondominated: list[EvaluationResult] = []
    for candidate in results:
        if not any(
            other is not candidate and _dominates(other, candidate, cohort_keys)
            for other in results
        ):
            nondominated.append(candidate)
    return tuple(sorted(nondominated, key=lambda item: _reference_key(item.candidate)))


def _dominates(
    left: EvaluationResult,
    right: EvaluationResult,
    cohort_keys: dict[tuple[str, str, str], tuple[int, ...]],
) -> bool:
    left_key = cohort_keys[_reference_key(left.candidate)]
    right_key = cohort_keys[_reference_key(right.candidate)]
    return all(
        left <= right for left, right in zip(left_key, right_key, strict=True)
    ) and (any(left < right for left, right in zip(left_key, right_key, strict=True)))
