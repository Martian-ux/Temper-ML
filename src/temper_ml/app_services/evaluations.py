"""Store-backed Slice 6 evaluation, recommendation, and review services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import re
import secrets
from typing import Any

from temper_ml.app_services._records import (
    require_no_conflicting_logical_revision,
    write_record_idempotently,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.artifacts import Artifact
from temper_ml.domain.evaluations import (
    ArtifactIntegrityStatus,
    BaselineOutcome,
    CaseSuiteKind,
    EvaluationCase,
    EvaluationResult,
    EvaluationSuite,
    EvidenceStatus,
    MetricDirection,
    Recommendation,
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
    build_recommendation,
)
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
    record_reference,
    require_identifier,
    require_text,
    thaw_json,
)
from temper_ml.domain.runs import EvaluationMode
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest
from temper_ml.store.redaction import (
    PublicSafetyError,
    RedactionContext,
    validate_canonical_admission,
)

BLIND_PACKET_PROJECTION = HashProjection("review.blind_packet", "v1")
BLIND_MAPPING_PROJECTION = HashProjection("review.blind_mapping", "v1")


@dataclass(frozen=True)
class BlindCandidateOutput:
    """An unblinded service input that is never canonical evidence itself."""

    candidate: RecordReference
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate, RecordReference)
            or self.candidate.record_type != "artifact"
        ):
            raise RecordValidationError(
                "blind output candidate must reference artifact"
            )
        object.__setattr__(
            self,
            "output",
            freeze_json_object(self.output, field="blind candidate output"),
        )


@dataclass(frozen=True)
class BlindReviewInput:
    """One synchronized prompt and its candidate-labelled source outputs."""

    prompt_id: str
    prompt: Mapping[str, Any]
    settings: Mapping[str, Any]
    outputs: tuple[BlindCandidateOutput, ...]

    def __post_init__(self) -> None:
        require_identifier("blind review prompt_id", self.prompt_id)
        object.__setattr__(
            self,
            "prompt",
            freeze_json_object(self.prompt, field="blind review prompt"),
        )
        object.__setattr__(
            self,
            "settings",
            freeze_json_object(self.settings, field="blind review settings"),
        )
        if (
            not isinstance(self.outputs, tuple)
            or not self.outputs
            or any(
                not isinstance(output, BlindCandidateOutput) for output in self.outputs
            )
        ):
            raise RecordValidationError("blind review outputs must be non-empty")
        keys = tuple(_reference_key(output.candidate) for output in self.outputs)
        if len(set(keys)) != len(keys):
            raise RecordValidationError("blind review candidates must be unique")


@dataclass(frozen=True)
class BlindPacketEntry:
    prompt_id: str
    prompt: Mapping[str, Any]
    settings: Mapping[str, Any]
    outputs: tuple[ReviewOutput, ...]

    def __post_init__(self) -> None:
        require_identifier("blind packet prompt_id", self.prompt_id)
        object.__setattr__(
            self,
            "prompt",
            freeze_json_object(self.prompt, field="blind packet prompt"),
        )
        object.__setattr__(
            self,
            "settings",
            freeze_json_object(self.settings, field="blind packet settings"),
        )
        if (
            not isinstance(self.outputs, tuple)
            or not self.outputs
            or any(not isinstance(output, ReviewOutput) for output in self.outputs)
        ):
            raise RecordValidationError("blind packet outputs must be non-empty")
        ordered = tuple(sorted(self.outputs, key=lambda output: output.alias))
        if len({output.alias for output in ordered}) != len(ordered):
            raise RecordValidationError("blind packet aliases must be unique")
        object.__setattr__(self, "outputs", ordered)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": thaw_json(self.prompt),
            "settings": thaw_json(self.settings),
            "outputs": [output.to_dict() for output in self.outputs],
        }


@dataclass(frozen=True)
class BlindReviewPacket:
    """Leak-audited public packet with only a salted mapping commitment."""

    packet_id: str
    entries: tuple[BlindPacketEntry, ...]
    mapping_commitment: ContentIdentity
    packet_identity: ContentIdentity
    leak_audit_passed: bool

    def __post_init__(self) -> None:
        require_identifier("blind packet_id", self.packet_id)
        if (
            not isinstance(self.entries, tuple)
            or not self.entries
            or any(not isinstance(entry, BlindPacketEntry) for entry in self.entries)
        ):
            raise RecordValidationError("blind packet entries must be non-empty")
        ordered_entries = tuple(sorted(self.entries, key=lambda entry: entry.prompt_id))
        if len({entry.prompt_id for entry in ordered_entries}) != len(ordered_entries):
            raise RecordValidationError("blind packet prompt ids must be unique")
        aliases = tuple(output.alias for output in ordered_entries[0].outputs)
        if any(
            tuple(output.alias for output in entry.outputs) != aliases
            for entry in ordered_entries
        ):
            raise RecordValidationError("blind packet entries must share aliases")
        object.__setattr__(self, "entries", ordered_entries)
        if not isinstance(self.mapping_commitment, ContentIdentity):
            raise RecordValidationError("blind mapping commitment is invalid")
        if not isinstance(self.packet_identity, ContentIdentity):
            raise RecordValidationError("blind packet identity is invalid")
        if self.packet_identity != _blind_packet_identity(
            self.packet_id,
            ordered_entries,
            self.mapping_commitment,
        ):
            raise RecordValidationError("blind packet identity mismatch")
        if self.leak_audit_passed is not True:
            raise RecordValidationError("blind packet must pass its leak audit")

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(output.alias for output in self.entries[0].outputs)

    def public_fields(self) -> dict[str, object]:
        """Return the exact identity-bound packet without candidate identities."""

        return {
            "packet_id": self.packet_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "mapping_commitment": {
                "algorithm": self.mapping_commitment.algorithm,
                "value": self.mapping_commitment.value,
            },
        }


@dataclass(frozen=True)
class BlindReviewJudgment:
    prompt_id: str
    notes: str
    ratings: tuple[ReviewRating, ...]

    def __post_init__(self) -> None:
        require_identifier("blind judgment prompt_id", self.prompt_id)
        require_text("blind judgment notes", self.notes)
        if (
            not isinstance(self.ratings, tuple)
            or not self.ratings
            or any(not isinstance(rating, ReviewRating) for rating in self.ratings)
        ):
            raise RecordValidationError("blind judgment ratings must be non-empty")


class EvaluationService:
    """Canonical Slice 6 workflow with strict reference and revision validation."""

    def __init__(self, project_root: Path | str) -> None:
        self.store = TypedEvidenceStore(project_root)

    def register_suite(self, suite: EvaluationSuite) -> EvaluationSuite:
        if not isinstance(suite, EvaluationSuite):
            raise ApplicationServiceError("evaluation_suite_invalid")
        if (
            suite.state is not SuiteEvidenceState.UNSEALED
            or suite.prior_suite is not None
        ):
            raise ApplicationServiceError("evaluation_suite_initial_state_invalid")
        self._persist_new(
            suite,
            conflict_code="evaluation_suite_revision_conflict",
        )
        self._append(
            f"evaluation-suite-{suite.suite_id}",
            f"suite-registered-{suite.identity.value}",
            "evaluation_suite_registered",
            {
                "suite_kind": suite.kind.value,
                "suite_state": suite.state.value,
                "case_count": len(suite.cases),
                "evaluator_count": len(suite.evaluators),
            },
        )
        self.store.verify()
        return suite

    def seal_suite(self, suite: EvaluationSuite) -> EvaluationSuite:
        current = self._require_suite_revision(suite)
        if (
            current.state is not SuiteEvidenceState.UNSEALED
            or current.prior_suite is not None
        ):
            raise ApplicationServiceError("evaluation_suite_cannot_seal")
        return self._write_suite_revision(current, SuiteEvidenceState.SEALED)

    def inspect_confirmation_suite(self, suite: EvaluationSuite) -> EvaluationSuite:
        current = self._require_suite_revision(suite)
        if (
            current.kind is not CaseSuiteKind.CONFIRMATION
            or current.state is not SuiteEvidenceState.SEALED
        ):
            raise ApplicationServiceError("confirmation_suite_not_sealed")
        return self._write_suite_revision(current, SuiteEvidenceState.UNSEALED)

    def modify_suite(
        self,
        suite: EvaluationSuite,
        *,
        cases: tuple[EvaluationCase, ...],
    ) -> EvaluationSuite:
        current = self._require_suite_revision(suite)
        if current.state is SuiteEvidenceState.CONTAMINATED:
            raise ApplicationServiceError("evaluation_suite_contaminated")
        if current.state is SuiteEvidenceState.RETIRED:
            raise ApplicationServiceError("evaluation_suite_retired")
        try:
            revised = replace(
                current,
                state=SuiteEvidenceState.MODIFIED,
                cases=cases,
                prior_suite=record_reference(current),
            )
        except (RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError("evaluation_suite_cases_invalid") from None
        if revised.case_membership_identity == current.case_membership_identity:
            raise ApplicationServiceError("evaluation_suite_modification_empty")
        return self._persist_suite_revision(current, revised)

    def mark_suite_contaminated(self, suite: EvaluationSuite) -> EvaluationSuite:
        current = self._require_suite_revision(suite)
        if current.state is SuiteEvidenceState.RETIRED:
            raise ApplicationServiceError("evaluation_suite_retired")
        if current.state is SuiteEvidenceState.CONTAMINATED:
            return self._require_current_suite(current)
        return self._write_suite_revision(current, SuiteEvidenceState.CONTAMINATED)

    def retire_suite(self, suite: EvaluationSuite) -> EvaluationSuite:
        current = self._require_suite_revision(suite)
        if current.state is SuiteEvidenceState.RETIRED:
            return self._require_current_suite(current)
        return self._write_suite_revision(current, SuiteEvidenceState.RETIRED)

    def record_solo_review(self, review: Review) -> Review:
        if (
            not isinstance(review, Review)
            or review.mode is not ReviewMode.SOLO
            or review.stage is not ReviewStage.RECORDED
        ):
            raise ApplicationServiceError("solo_review_invalid")
        for mapping in review.candidate_mappings:
            self._require_reference(mapping.candidate, "artifact")
        self._persist_new(review, conflict_code="review_revision_conflict")
        self._append(
            f"review-{review.review_id}",
            f"solo-review-recorded-{review.identity.value}",
            "solo_review_recorded",
            {
                "structured_review": True,
                "prompt_count": len(review.entries),
                "candidate_count": len(review.candidate_mappings),
            },
        )
        self.store.verify()
        return review

    def prepare_blind_review(
        self,
        packet_id: str,
        inputs: tuple[BlindReviewInput, ...],
    ) -> BlindReviewPacket:
        try:
            require_identifier("blind packet_id", packet_id)
        except RecordValidationError:
            raise ApplicationServiceError("blind_review_packet_invalid") from None
        if (
            not isinstance(inputs, tuple)
            or not inputs
            or any(not isinstance(item, BlindReviewInput) for item in inputs)
        ):
            raise ApplicationServiceError("blind_review_inputs_invalid")
        ordered_inputs = tuple(sorted(inputs, key=lambda item: item.prompt_id))
        if len({item.prompt_id for item in ordered_inputs}) != len(ordered_inputs):
            raise ApplicationServiceError("blind_review_prompt_conflict")
        candidate_keys = {
            _reference_key(output.candidate) for output in ordered_inputs[0].outputs
        }
        if any(
            {_reference_key(output.candidate) for output in item.outputs}
            != candidate_keys
            for item in ordered_inputs
        ):
            raise ApplicationServiceError("blind_review_candidate_set_mismatch")
        candidates = tuple(
            sorted(
                (output.candidate for output in ordered_inputs[0].outputs),
                key=_reference_key,
            )
        )
        for candidate in candidates:
            self._require_reference(candidate, "artifact")
        existing = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, Review)
            and stored.record.review_id == packet_id
        )
        if existing:
            preparation = self._require_blind_preparation(packet_id)
            mappings = preparation.candidate_mappings
            if {
                _reference_key(mapping.candidate) for mapping in mappings
            } != candidate_keys:
                raise ApplicationServiceError("blind_review_candidate_set_mismatch")
            hiding_nonce = preparation.hiding_nonce
            if hiding_nonce is None:
                raise ApplicationServiceError("blind_review_preparation_invalid")
        else:
            shuffled = list(candidates)
            secrets.SystemRandom().shuffle(shuffled)
            mappings = tuple(
                ReviewCandidate(f"candidate-{index:03d}", candidate)
                for index, candidate in enumerate(shuffled, 1)
            )
            hiding_nonce = secrets.token_hex(32)
        entries = _blind_packet_entries(ordered_inputs, mappings)
        commitment = _blind_mapping_commitment(hiding_nonce, mappings)
        packet = BlindReviewPacket(
            packet_id=packet_id,
            entries=entries,
            mapping_commitment=commitment,
            packet_identity=_blind_packet_identity(packet_id, entries, commitment),
            leak_audit_passed=True,
        )
        _audit_blind_packet(packet, mappings)
        try:
            validate_canonical_admission(
                packet.public_fields(),
                context=RedactionContext.current(),
            )
        except PublicSafetyError:
            raise ApplicationServiceError(
                "blind_review_packet_public_safety_failed"
            ) from None
        if existing:
            if preparation.packet_identity != packet.packet_identity:
                raise ApplicationServiceError("blind_review_packet_conflict")
        else:
            try:
                preparation = Review(
                    review_id=packet_id,
                    mode=ReviewMode.BLIND,
                    stage=ReviewStage.BLIND_PREPARED,
                    entries=(),
                    reviewer_declaration=None,
                    candidate_mappings=mappings,
                    leak_audit_passed=True,
                    packet_identity=packet.packet_identity,
                    hiding_nonce=hiding_nonce,
                )
            except (RecordValidationError, TypeError, ValueError):
                raise ApplicationServiceError(
                    "blind_review_preparation_invalid"
                ) from None
            self._persist_new(preparation, conflict_code="review_revision_conflict")
        self._append(
            f"review-{packet_id}",
            f"blind-review-prepared-{preparation.identity.value}",
            "blind_review_prepared",
            {
                "leak_audit_passed": True,
                "judgment_sealed": False,
                "identities_revealed": False,
                "prompt_count": len(entries),
                "candidate_count": len(mappings),
            },
        )
        self.store.verify()
        return packet

    def seal_blind_review(
        self,
        review_id: str,
        packet: BlindReviewPacket,
        judgments: tuple[BlindReviewJudgment, ...],
        *,
        reviewer_declaration: str,
    ) -> Review:
        if not isinstance(packet, BlindReviewPacket) or not packet.leak_audit_passed:
            raise ApplicationServiceError("blind_review_packet_invalid")
        if review_id != packet.packet_id:
            raise ApplicationServiceError("blind_review_id_mismatch")
        preparation = self._require_blind_preparation(packet.packet_id, packet)
        mappings = preparation.candidate_mappings
        _audit_blind_packet(packet, mappings)
        for mapping in mappings:
            self._require_reference(mapping.candidate, "artifact")
        if (
            not isinstance(judgments, tuple)
            or not judgments
            or any(
                not isinstance(judgment, BlindReviewJudgment) for judgment in judgments
            )
        ):
            raise ApplicationServiceError("blind_review_judgments_invalid")
        by_prompt = {judgment.prompt_id: judgment for judgment in judgments}
        if len(by_prompt) != len(judgments) or set(by_prompt) != {
            entry.prompt_id for entry in packet.entries
        }:
            raise ApplicationServiceError("blind_review_judgment_set_mismatch")
        try:
            entries = tuple(
                ReviewEntry(
                    prompt_id=entry.prompt_id,
                    prompt=entry.prompt,
                    settings=entry.settings,
                    outputs=entry.outputs,
                    notes=by_prompt[entry.prompt_id].notes,
                    ratings=by_prompt[entry.prompt_id].ratings,
                )
                for entry in packet.entries
            )
            review = Review(
                review_id=review_id,
                mode=ReviewMode.BLIND,
                stage=ReviewStage.BLIND_SEALED,
                entries=entries,
                reviewer_declaration=reviewer_declaration,
                candidate_mappings=(),
                leak_audit_passed=True,
                packet_identity=packet.packet_identity,
                prior_review=record_reference(preparation),
            )
        except (RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError("blind_review_judgments_invalid") from None
        return self._persist_review_revision(
            preparation,
            review,
            event_key=f"blind-review-sealed-{review.identity.value}",
            event_type="blind_review_judgment_sealed",
            event_payload={
                "leak_audit_passed": True,
                "judgment_sealed": True,
                "identities_revealed": False,
                "prompt_count": len(review.entries),
            },
        )

    def reveal_blind_review(
        self,
        sealed_review: Review,
        packet: BlindReviewPacket,
    ) -> Review:
        if (
            not isinstance(sealed_review, Review)
            or sealed_review.mode is not ReviewMode.BLIND
            or sealed_review.stage is not ReviewStage.BLIND_SEALED
            or not isinstance(packet, BlindReviewPacket)
        ):
            raise ApplicationServiceError("blind_review_not_sealed")
        preparation = self._require_blind_preparation(packet.packet_id, packet)
        mappings = preparation.candidate_mappings
        _audit_blind_packet(packet, mappings)
        sealed_review = self._require_review_revision(sealed_review)
        if not _sealed_review_matches_preparation(sealed_review, preparation):
            raise ApplicationServiceError("blind_review_packet_mismatch")
        if sealed_review.packet_identity != packet.packet_identity:
            raise ApplicationServiceError("blind_review_packet_mismatch")
        if sealed_review.review_id != packet.packet_id:
            raise ApplicationServiceError("blind_review_id_mismatch")
        for mapping in mappings:
            self._require_reference(mapping.candidate, "artifact")
        try:
            revealed = replace(
                sealed_review,
                stage=ReviewStage.BLIND_REVEALED,
                candidate_mappings=mappings,
                prior_review=record_reference(sealed_review),
            )
        except (RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError("blind_review_reveal_invalid") from None
        return self._persist_review_revision(
            sealed_review,
            revealed,
            event_key=f"blind-review-revealed-{revealed.identity.value}",
            event_type="blind_review_identities_revealed",
            event_payload={
                "judgment_sealed": True,
                "identities_revealed": True,
                "candidate_count": len(revealed.candidate_mappings),
            },
        )

    def record_result(self, result: EvaluationResult) -> EvaluationResult:
        if not isinstance(result, EvaluationResult):
            raise ApplicationServiceError("evaluation_result_invalid")
        suite_backed_modes = (
            EvaluationMode.FULL_SUITE,
            EvaluationMode.EXPERIMENT_LOOP,
        )
        if result.evaluation_mode in suite_backed_modes and result.suite is None:
            raise ApplicationServiceError("evaluation_suite_required_for_mode")
        if (
            result.evaluation_mode not in suite_backed_modes
            and result.suite is not None
        ):
            raise ApplicationServiceError("evaluation_suite_mode_mismatch")
        candidate_record = self._require_reference(result.candidate, "artifact")
        if not isinstance(candidate_record, Artifact):
            raise ApplicationServiceError("evaluation_candidate_reference_invalid")
        if (
            result.artifact_integrity_status is ArtifactIntegrityStatus.PASSED
            and result.artifact_integrity_evidence
            != candidate_record.integrity_evidence
        ):
            raise ApplicationServiceError("artifact_integrity_evidence_mismatch")
        suite: EvaluationSuite | None = None
        if result.suite is not None:
            suite_record = self._require_reference(result.suite, "evaluation_suite")
            if not isinstance(suite_record, EvaluationSuite):
                raise ApplicationServiceError("evaluation_suite_reference_invalid")
            suite = suite_record
            self._require_current_suite(suite)
            if result.suite_state is not suite.state:
                raise ApplicationServiceError("evaluation_suite_state_mismatch")
            if suite.state is SuiteEvidenceState.RETIRED:
                raise ApplicationServiceError("evaluation_suite_retired")
            if (
                result.evaluation_mode is EvaluationMode.EXPERIMENT_LOOP
                and suite.kind is CaseSuiteKind.CONFIRMATION
            ):
                raise ApplicationServiceError(
                    "experiment_loop_confirmation_suite_invalid"
                )
            if (
                suite.kind is CaseSuiteKind.CONFIRMATION
                and suite.state
                in (
                    SuiteEvidenceState.UNSEALED,
                    SuiteEvidenceState.MODIFIED,
                    SuiteEvidenceState.CONTAMINATED,
                )
                and result.evidence_status is not EvidenceStatus.CONTAMINATED
            ):
                raise ApplicationServiceError("confirmation_evidence_contaminated")
            evaluator_by_metric = {
                evaluator.metric_name: evaluator for evaluator in suite.evaluators
            }
            for metric in result.metrics:
                evaluator = evaluator_by_metric.get(metric.metric_name)
                if (
                    evaluator is None
                    or evaluator.kind is not metric.evaluator_kind
                    or evaluator.direction is not metric.direction
                ):
                    raise ApplicationServiceError("evaluation_metric_not_declared")
        if result.review is not None:
            review_record = self._require_reference(result.review, "review")
            if not isinstance(review_record, Review):
                raise ApplicationServiceError("evaluation_review_reference_invalid")
            self._require_current_review(review_record)
            if review_record.mode is ReviewMode.BLIND and (
                review_record.stage is not ReviewStage.BLIND_REVEALED
            ):
                raise ApplicationServiceError("blind_review_identity_not_revealed")
            candidates = {
                _reference_key(mapping.candidate)
                for mapping in review_record.candidate_mappings
            }
            if _reference_key(result.candidate) not in candidates:
                raise ApplicationServiceError("evaluation_review_candidate_mismatch")
        candidate_metrics = {metric.metric_name: metric for metric in result.metrics}
        for comparison in result.baseline_comparisons:
            baseline_record = self._require_reference(
                comparison.baseline, "evaluation_result"
            )
            if not isinstance(baseline_record, EvaluationResult):
                raise ApplicationServiceError("baseline_evidence_invalid")
            if baseline_record.candidate == result.candidate:
                raise ApplicationServiceError("baseline_candidate_self_reference")
            if baseline_record.evaluation_mode is not result.evaluation_mode:
                raise ApplicationServiceError("baseline_evaluation_mode_mismatch")
            if (
                result.suite is None
                or result.suite_state is None
                or baseline_record.suite != result.suite
                or baseline_record.suite_state is not result.suite_state
            ):
                raise ApplicationServiceError("baseline_suite_context_mismatch")
            baseline_candidate = self._require_reference(
                baseline_record.candidate,
                "artifact",
            )
            if not isinstance(baseline_candidate, Artifact):
                raise ApplicationServiceError("baseline_candidate_reference_invalid")
            if (
                baseline_record.artifact_integrity_status
                is not ArtifactIntegrityStatus.PASSED
            ):
                raise ApplicationServiceError("baseline_artifact_integrity_invalid")
            if (
                baseline_record.artifact_integrity_evidence
                != baseline_candidate.integrity_evidence
            ):
                raise ApplicationServiceError(
                    "baseline_artifact_integrity_evidence_mismatch"
                )
            if baseline_record.evidence_status not in (
                EvidenceStatus.PASSED,
                EvidenceStatus.FAILED,
            ):
                raise ApplicationServiceError("baseline_evidence_status_invalid")
            candidate_metric = candidate_metrics.get(comparison.metric_name)
            baseline_metric = next(
                (
                    metric
                    for metric in baseline_record.metrics
                    if metric.metric_name == comparison.metric_name
                ),
                None,
            )
            if candidate_metric is None or baseline_metric is None:
                raise ApplicationServiceError("baseline_metric_missing")
            if comparison.candidate_value != candidate_metric.value:
                raise ApplicationServiceError("baseline_candidate_value_mismatch")
            if comparison.baseline_value != baseline_metric.value:
                raise ApplicationServiceError("baseline_value_mismatch")
            if (
                candidate_metric.direction is not baseline_metric.direction
                or candidate_metric.evaluator_kind is not baseline_metric.evaluator_kind
            ):
                raise ApplicationServiceError("baseline_metric_semantics_mismatch")
            expected_outcome = BaselineOutcome.EQUIVALENT
            if candidate_metric.value != baseline_metric.value:
                candidate_is_better = (
                    candidate_metric.value > baseline_metric.value
                    if candidate_metric.direction is MetricDirection.MAXIMIZE
                    else candidate_metric.value < baseline_metric.value
                )
                expected_outcome = (
                    BaselineOutcome.BETTER
                    if candidate_is_better
                    else BaselineOutcome.WORSE
                )
            if (
                comparison.outcome is not BaselineOutcome.INCONCLUSIVE
                and comparison.outcome is not expected_outcome
            ):
                raise ApplicationServiceError("baseline_outcome_mismatch")
        self._persist_new(result, conflict_code="evaluation_result_revision_conflict")
        self._append(
            "evaluation-results",
            f"evaluation-result-recorded-{result.identity.value}",
            "evaluation_result_recorded",
            {
                "evaluation_mode": result.evaluation_mode.value,
                "artifact_integrity_status": result.artifact_integrity_status.value,
                "evidence_status": result.evidence_status.value,
                "metric_count": len(result.metrics),
                "baseline_comparison_count": len(result.baseline_comparisons),
                "suite_state_disclosed": suite is not None,
            },
        )
        self.store.verify()
        return result

    def register_policy(self, policy: RecommendationPolicy) -> RecommendationPolicy:
        if not isinstance(policy, RecommendationPolicy):
            raise ApplicationServiceError("recommendation_policy_invalid")
        self._persist_new(policy, conflict_code="recommendation_policy_conflict")
        self._append(
            "recommendation-policies",
            f"recommendation-policy-recorded-{policy.identity.value}",
            "recommendation_policy_recorded",
            {
                "hard_qualifier_count": len(policy.hard_qualifiers),
                "advisory_metric_count": len(policy.advisory_metrics),
                "objective_count": len(policy.objectives),
                "baseline_comparison_count": len(policy.baseline_comparisons),
            },
        )
        self.store.verify()
        return policy

    def recommend(
        self,
        recommendation_id: str,
        policy: RecommendationPolicy,
        results: tuple[EvaluationResult, ...],
    ) -> Recommendation:
        self._require_exact_record(policy, "recommendation_policy_store_mismatch")
        if not isinstance(results, tuple) or not results:
            raise ApplicationServiceError("recommendation_results_invalid")
        for result in results:
            if not isinstance(result, EvaluationResult):
                raise ApplicationServiceError("recommendation_results_invalid")
            self._require_exact_record(result, "evaluation_result_store_mismatch")
            self._require_reference(result.candidate, "artifact")
            if result.suite is not None:
                suite_record = self._require_reference(result.suite, "evaluation_suite")
                if not isinstance(suite_record, EvaluationSuite):
                    raise ApplicationServiceError("evaluation_suite_reference_invalid")
                self._require_current_suite(suite_record)
        try:
            recommendation = build_recommendation(
                recommendation_id,
                policy,
                results,
            )
        except (RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError(
                "recommendation_policy_evaluation_failed"
            ) from None
        self._persist_new(
            recommendation,
            conflict_code="recommendation_revision_conflict",
        )
        self._append(
            "recommendations",
            f"recommendation-recorded-{recommendation.identity.value}",
            "recommendation_recorded",
            {
                "candidate_count": len(recommendation.assessments),
                "qualified_count": sum(
                    assessment.qualified for assessment in recommendation.assessments
                ),
                "selection_available": recommendation.selected_candidate is not None,
                "confidence": recommendation.confidence.value,
                "conflict_count": len(recommendation.conflicts),
            },
        )
        self.store.verify()
        return recommendation

    def record_decision(self, decision: UserDecision) -> UserDecision:
        if not isinstance(decision, UserDecision):
            raise ApplicationServiceError("user_decision_invalid")
        recommendation_record = self._require_reference(
            decision.recommendation, "recommendation"
        )
        if not isinstance(recommendation_record, Recommendation):
            raise ApplicationServiceError("recommendation_reference_invalid")
        assessment = next(
            (
                item
                for item in recommendation_record.assessments
                if item.candidate == decision.candidate
            ),
            None,
        )
        if assessment is None:
            raise ApplicationServiceError("decision_candidate_not_recommended")
        if assessment.evidence_status is not decision.evidence_status_at_decision:
            raise ApplicationServiceError("decision_evidence_status_mismatch")
        self._require_reference(decision.candidate, "artifact")
        self._persist_new(decision, conflict_code="user_decision_revision_conflict")
        warned = decision.evidence_status_at_decision is not EvidenceStatus.PASSED
        self._append(
            "user-decisions",
            f"user-decision-recorded-{decision.identity.value}",
            "user_decision_recorded",
            {
                "decision_status": decision.status.value,
                "evidence_status_unchanged": decision.evidence_status_at_decision.value,
                "warned_candidate": warned,
                "override_reason_recorded": decision.override_reason is not None,
            },
        )
        self.store.verify()
        return decision

    def decision_history(self) -> tuple[UserDecision, ...]:
        """Return immutable registry decisions in their append-only event order."""

        decisions = {
            record.identity.value: record
            for stored in self.store.iter_records()
            if isinstance((record := stored.record), UserDecision)
        }
        snapshots = [
            snapshot
            for snapshot in self.store.iter_streams()
            if snapshot.stream_id == "user-decisions"
        ]
        if not snapshots:
            if decisions:
                raise ApplicationServiceError("user_decision_event_missing")
            return ()
        if len(snapshots) != 1:
            raise ApplicationServiceError("user_decision_stream_ambiguous")
        prefix = "user-decision-recorded-"
        history: list[UserDecision] = []
        for event in snapshots[0].events:
            if event.event_type != "user_decision_recorded":
                raise ApplicationServiceError("user_decision_event_invalid")
            if not event.idempotency_key.startswith(prefix):
                raise ApplicationServiceError("user_decision_event_invalid")
            identity = event.idempotency_key.removeprefix(prefix)
            decision = decisions.pop(identity, None)
            if decision is None:
                raise ApplicationServiceError("user_decision_event_invalid")
            history.append(decision)
        if decisions:
            raise ApplicationServiceError("user_decision_event_missing")
        return tuple(history)

    def current_decision(self) -> UserDecision | None:
        """Resolve the one current registry state; each later event supersedes it."""

        history = self.decision_history()
        return history[-1] if history else None

    def _persist_new(self, record: TypedRecord, *, conflict_code: str) -> None:
        try:
            require_no_conflicting_logical_revision(
                self.store,
                record,
                conflict_code=conflict_code,
            )
            write_record_idempotently(
                self.store,
                record,
                conflict_code=conflict_code,
            )
        except ApplicationServiceError:
            raise
        except (EvidenceError, RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError(conflict_code) from None

    def _write_suite_revision(
        self,
        current: EvaluationSuite,
        state: SuiteEvidenceState,
    ) -> EvaluationSuite:
        try:
            revised = replace(
                current,
                state=state,
                prior_suite=record_reference(current),
            )
        except (RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError(
                "evaluation_suite_transition_invalid"
            ) from None
        return self._persist_suite_revision(current, revised)

    def _persist_suite_revision(
        self,
        current: EvaluationSuite,
        revised: EvaluationSuite,
    ) -> EvaluationSuite:
        if revised.prior_suite != record_reference(current):
            raise ApplicationServiceError("evaluation_suite_lineage_invalid")
        current = self._require_suite_revision(current)
        revisions, successors, head = self._validated_suite_lineage(current.suite_id)
        del revisions
        successor = successors.get(record_reference(current))
        if successor is not None:
            if successor.to_dict() != revised.to_dict():
                raise ApplicationServiceError("evaluation_suite_revision_conflict")
            persisted = successor
        else:
            if record_reference(head) != record_reference(current):
                raise ApplicationServiceError("evaluation_suite_revision_stale")
            try:
                write_record_idempotently(
                    self.store,
                    revised,
                    conflict_code="evaluation_suite_revision_conflict",
                )
                self._validated_suite_lineage(current.suite_id)
            except (ApplicationServiceError, EvidenceError, RecordValidationError):
                raise ApplicationServiceError(
                    "evaluation_suite_revision_conflict"
                ) from None
            persisted = revised
        self._append(
            f"evaluation-suite-{persisted.suite_id}",
            f"suite-state-{persisted.identity.value}",
            "evaluation_suite_state_changed",
            {
                "prior_state": current.state.value,
                "suite_state": persisted.state.value,
                "case_membership_changed": (
                    persisted.case_membership_identity
                    != current.case_membership_identity
                ),
            },
        )
        self.store.verify()
        return persisted

    def _validated_suite_lineage(
        self,
        suite_id: str,
    ) -> tuple[
        dict[RecordReference, EvaluationSuite],
        dict[RecordReference, EvaluationSuite],
        EvaluationSuite,
    ]:
        revisions = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, EvaluationSuite)
            and stored.record.suite_id == suite_id
        )
        if not revisions:
            raise ApplicationServiceError("evaluation_suite_store_mismatch")
        by_reference = {record_reference(revision): revision for revision in revisions}
        roots = tuple(
            revision for revision in revisions if revision.prior_suite is None
        )
        if len(by_reference) != len(revisions) or len(roots) != 1:
            raise ApplicationServiceError("evaluation_suite_revision_conflict")
        root = roots[0]
        successors: dict[RecordReference, EvaluationSuite] = {}
        for revision in revisions:
            prior = revision.prior_suite
            if prior is None:
                continue
            if prior not in by_reference or revision.kind is not root.kind:
                raise ApplicationServiceError("evaluation_suite_revision_conflict")
            if prior in successors:
                raise ApplicationServiceError("evaluation_suite_revision_conflict")
            successors[prior] = revision
        heads = tuple(
            revision
            for reference, revision in by_reference.items()
            if reference not in successors
        )
        if len(heads) != 1:
            raise ApplicationServiceError("evaluation_suite_revision_conflict")
        visited: set[RecordReference] = set()
        cursor = root
        while record_reference(cursor) not in visited:
            cursor_reference = record_reference(cursor)
            visited.add(cursor_reference)
            successor = successors.get(cursor_reference)
            if successor is None:
                break
            cursor = successor
        if len(visited) != len(revisions) or cursor != heads[0]:
            raise ApplicationServiceError("evaluation_suite_revision_conflict")
        return by_reference, successors, heads[0]

    def _require_suite_revision(self, suite: EvaluationSuite) -> EvaluationSuite:
        if not isinstance(suite, EvaluationSuite):
            raise ApplicationServiceError("evaluation_suite_invalid")
        self._require_exact_record(suite, "evaluation_suite_store_mismatch")
        revisions, _, _ = self._validated_suite_lineage(suite.suite_id)
        stored = revisions.get(record_reference(suite))
        if stored is None or stored.to_dict() != suite.to_dict():
            raise ApplicationServiceError("evaluation_suite_store_mismatch")
        return stored

    def _require_current_suite(self, suite: EvaluationSuite) -> EvaluationSuite:
        stored = self._require_suite_revision(suite)
        _, _, head = self._validated_suite_lineage(stored.suite_id)
        if record_reference(head) != record_reference(stored):
            raise ApplicationServiceError("evaluation_suite_revision_stale")
        return stored

    def _persist_review_revision(
        self,
        current: Review,
        revised: Review,
        *,
        event_key: str,
        event_type: str,
        event_payload: Mapping[str, Any],
    ) -> Review:
        if revised.prior_review != record_reference(current):
            raise ApplicationServiceError("review_lineage_invalid")
        current = self._require_review_revision(current)
        _, successors, head = self._validated_review_lineage(current.review_id)
        successor = successors.get(record_reference(current))
        if successor is not None:
            if successor.to_dict() != revised.to_dict():
                raise ApplicationServiceError("review_revision_conflict")
            persisted = successor
        else:
            if record_reference(head) != record_reference(current):
                raise ApplicationServiceError("review_revision_stale")
            try:
                write_record_idempotently(
                    self.store,
                    revised,
                    conflict_code="review_revision_conflict",
                )
                self._validated_review_lineage(current.review_id)
            except (ApplicationServiceError, EvidenceError, RecordValidationError):
                raise ApplicationServiceError("review_revision_conflict") from None
            persisted = revised
        self._append(
            f"review-{persisted.review_id}",
            event_key,
            event_type,
            event_payload,
        )
        self.store.verify()
        return persisted

    def _validated_review_lineage(
        self,
        review_id: str,
    ) -> tuple[
        dict[RecordReference, Review],
        dict[RecordReference, Review],
        Review,
    ]:
        revisions = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, Review)
            and stored.record.review_id == review_id
        )
        if not revisions:
            raise ApplicationServiceError("review_store_mismatch")
        by_reference = {record_reference(revision): revision for revision in revisions}
        roots = tuple(
            revision for revision in revisions if revision.prior_review is None
        )
        if len(by_reference) != len(revisions) or len(roots) != 1:
            raise ApplicationServiceError("review_revision_conflict")
        root = roots[0]
        if (
            root.mode is ReviewMode.BLIND
            and root.stage is not ReviewStage.BLIND_PREPARED
        ):
            raise ApplicationServiceError("review_revision_conflict")
        successors: dict[RecordReference, Review] = {}
        allowed_transitions = {
            (ReviewStage.BLIND_PREPARED, ReviewStage.BLIND_SEALED),
            (ReviewStage.BLIND_SEALED, ReviewStage.BLIND_REVEALED),
        }
        for revision in revisions:
            prior = revision.prior_review
            if prior is None:
                continue
            predecessor = by_reference.get(prior)
            if (
                predecessor is None
                or revision.mode is not root.mode
                or (predecessor.stage, revision.stage) not in allowed_transitions
                or prior in successors
                or not _blind_review_transition_is_valid(
                    root,
                    predecessor,
                    revision,
                )
            ):
                raise ApplicationServiceError("review_revision_conflict")
            successors[prior] = revision
        heads = tuple(
            revision
            for reference, revision in by_reference.items()
            if reference not in successors
        )
        if len(heads) != 1:
            raise ApplicationServiceError("review_revision_conflict")
        visited: set[RecordReference] = set()
        cursor = root
        while record_reference(cursor) not in visited:
            cursor_reference = record_reference(cursor)
            visited.add(cursor_reference)
            successor = successors.get(cursor_reference)
            if successor is None:
                break
            cursor = successor
        if len(visited) != len(revisions) or cursor != heads[0]:
            raise ApplicationServiceError("review_revision_conflict")
        return by_reference, successors, heads[0]

    def _require_review_revision(self, review: Review) -> Review:
        if not isinstance(review, Review):
            raise ApplicationServiceError("review_invalid")
        self._require_exact_record(review, "review_store_mismatch")
        revisions, _, _ = self._validated_review_lineage(review.review_id)
        stored = revisions.get(record_reference(review))
        if stored is None or stored.to_dict() != review.to_dict():
            raise ApplicationServiceError("review_store_mismatch")
        return stored

    def _require_current_review(self, review: Review) -> Review:
        stored = self._require_review_revision(review)
        _, _, head = self._validated_review_lineage(stored.review_id)
        if record_reference(head) != record_reference(stored):
            raise ApplicationServiceError("review_revision_stale")
        return stored

    def _require_blind_preparation(
        self,
        review_id: str,
        packet: BlindReviewPacket | None = None,
    ) -> Review:
        revisions, _, _ = self._validated_review_lineage(review_id)
        preparations = tuple(
            revision
            for revision in revisions.values()
            if revision.mode is ReviewMode.BLIND
            and revision.stage is ReviewStage.BLIND_PREPARED
        )
        if len(preparations) != 1:
            raise ApplicationServiceError("blind_review_preparation_invalid")
        preparation = preparations[0]
        hiding_nonce = preparation.hiding_nonce
        if hiding_nonce is None:
            raise ApplicationServiceError("blind_review_preparation_invalid")
        if packet is not None and (
            packet.packet_id != review_id
            or preparation.packet_identity != packet.packet_identity
            or _blind_mapping_commitment(hiding_nonce, preparation.candidate_mappings)
            != packet.mapping_commitment
        ):
            raise ApplicationServiceError("blind_review_packet_mismatch")
        return preparation

    def _require_reference(
        self,
        reference: RecordReference,
        record_type: str,
    ) -> TypedRecord:
        if (
            not isinstance(reference, RecordReference)
            or reference.record_type != record_type
        ):
            raise ApplicationServiceError("record_reference_invalid")
        try:
            stored = self.store.read_record(reference)
        except (EvidenceError, RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError("record_reference_not_found") from None
        if record_reference(stored.record) != reference:
            raise ApplicationServiceError("record_reference_mismatch")
        return stored.record

    def _require_exact_record(self, record: TypedRecord, code: str) -> None:
        if not isinstance(record, TypedRecord):
            raise ApplicationServiceError(code)
        try:
            stored = self.store.read_record(record_reference(record))
        except (EvidenceError, RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError(code) from None
        if (
            type(stored.record) is not type(record)
            or stored.envelope.to_dict() != record.to_dict()
        ):
            raise ApplicationServiceError(code)

    def _append(
        self,
        stream_id: str,
        key: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        try:
            self.store.append_event(stream_id, EventRequest(key, event_type, payload))
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None


def _blind_packet_entries(
    inputs: tuple[BlindReviewInput, ...],
    mappings: tuple[ReviewCandidate, ...],
) -> tuple[BlindPacketEntry, ...]:
    alias_by_key = {
        _reference_key(mapping.candidate): mapping.alias for mapping in mappings
    }
    return tuple(
        BlindPacketEntry(
            prompt_id=item.prompt_id,
            prompt=item.prompt,
            settings=item.settings,
            outputs=tuple(
                ReviewOutput(
                    alias_by_key[_reference_key(output.candidate)],
                    output.output,
                )
                for output in item.outputs
            ),
        )
        for item in inputs
    )


def _blind_packet_identity(
    packet_id: str,
    entries: tuple[BlindPacketEntry, ...],
    mapping_commitment: ContentIdentity,
) -> ContentIdentity:
    return content_identity(
        BLIND_PACKET_PROJECTION,
        {
            "packet_id": packet_id,
            "entries": [entry.to_dict() for entry in entries],
            "mapping_commitment": {
                "algorithm": mapping_commitment.algorithm,
                "value": mapping_commitment.value,
            },
        },
    )


def _blind_mapping_commitment(
    hiding_nonce: str,
    mappings: tuple[ReviewCandidate, ...],
) -> ContentIdentity:
    return content_identity(
        BLIND_MAPPING_PROJECTION,
        {
            "schema_version": "v1",
            "hiding_nonce": hiding_nonce,
            "mappings": [mapping.to_dict() for mapping in mappings],
        },
    )


def _review_packet_entries(review: Review) -> tuple[BlindPacketEntry, ...]:
    return tuple(
        BlindPacketEntry(
            prompt_id=entry.prompt_id,
            prompt=entry.prompt,
            settings=entry.settings,
            outputs=entry.outputs,
        )
        for entry in review.entries
    )


def _sealed_review_matches_preparation(
    sealed: Review,
    preparation: Review,
) -> bool:
    hiding_nonce = preparation.hiding_nonce
    if (
        sealed.mode is not ReviewMode.BLIND
        or sealed.stage is not ReviewStage.BLIND_SEALED
        or preparation.mode is not ReviewMode.BLIND
        or preparation.stage is not ReviewStage.BLIND_PREPARED
        or hiding_nonce is None
        or sealed.review_id != preparation.review_id
        or sealed.packet_identity != preparation.packet_identity
    ):
        return False
    commitment = _blind_mapping_commitment(
        hiding_nonce,
        preparation.candidate_mappings,
    )
    return sealed.packet_identity == _blind_packet_identity(
        sealed.review_id,
        _review_packet_entries(sealed),
        commitment,
    )


def _blind_review_transition_is_valid(
    root: Review,
    predecessor: Review,
    revision: Review,
) -> bool:
    if (
        root.mode is not ReviewMode.BLIND
        or root.stage is not ReviewStage.BLIND_PREPARED
        or root.hiding_nonce is None
        or revision.review_id != root.review_id
        or revision.mode is not root.mode
        or revision.leak_audit_passed is not root.leak_audit_passed
        or revision.packet_identity != root.packet_identity
    ):
        return False
    committed_aliases = tuple(mapping.alias for mapping in root.candidate_mappings)
    if predecessor.stage is ReviewStage.BLIND_PREPARED:
        return (
            predecessor == root
            and revision.stage is ReviewStage.BLIND_SEALED
            and revision.candidate_mappings == ()
            and bool(revision.entries)
            and revision.entries[0].aliases == committed_aliases
            and _sealed_review_matches_preparation(revision, root)
        )
    if predecessor.stage is ReviewStage.BLIND_SEALED:
        return (
            revision.stage is ReviewStage.BLIND_REVEALED
            and revision.entries == predecessor.entries
            and revision.reviewer_declaration == predecessor.reviewer_declaration
            and revision.candidate_mappings == root.candidate_mappings
            and bool(revision.entries)
            and revision.entries[0].aliases == committed_aliases
            and _sealed_review_matches_preparation(predecessor, root)
        )
    return False


def _audit_blind_packet(
    packet: BlindReviewPacket,
    mappings: tuple[ReviewCandidate, ...],
) -> None:
    aliases = tuple(mapping.alias for mapping in mappings)
    if aliases != packet.aliases:
        raise ApplicationServiceError("blind_review_packet_invalid")
    if packet.packet_identity != _blind_packet_identity(
        packet.packet_id,
        packet.entries,
        packet.mapping_commitment,
    ):
        raise ApplicationServiceError("blind_review_packet_invalid")
    _audit_blind_fields(packet.public_fields(), mappings)


def _audit_blind_fields(
    public_fields: Mapping[str, Any],
    mappings: tuple[ReviewCandidate, ...],
) -> None:
    encoded = dumps_canonical_json(public_fields).decode()
    for mapping in mappings:
        reference = mapping.candidate
        identifier_pattern = (
            rf"(?<![A-Za-z0-9._-]){re.escape(reference.logical_id)}"
            r"(?![A-Za-z0-9._-])"
        )
        if (
            re.search(identifier_pattern, encoded) is not None
            or reference.identity.value in encoded
            or f"sha256:{reference.identity.value}" in encoded
        ):
            raise ApplicationServiceError("blind_review_leak_detected")


def _reference_key(reference: RecordReference) -> tuple[str, str, str]:
    return (reference.record_type, reference.logical_id, reference.identity.value)
