"""Application-service orchestration for the deterministic Slice 7 journey."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
from pathlib import Path
from typing import Any, Mapping

from temper_ml.app_services.datasets import (
    DatasetImportRequest,
    DatasetService,
    PreparedDataset,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.evaluations import (
    BlindCandidateOutput,
    BlindReviewInput,
    BlindReviewJudgment,
    BlindReviewPacket,
    EvaluationService,
)
from temper_ml.app_services.experiments import (
    ExperimentFreezeRequest,
    ExperimentService,
    ReplayMode,
    ReplayPlan,
    plan_replay,
    strict_replay_plan,
)
from temper_ml.app_services.local_use import (
    AdapterExportRequest,
    LocalUseRequest,
    LocalUseService,
)
from temper_ml.app_services.projects import (
    OpenedProject,
    ProjectCreateRequest,
    ProjectService,
)
from temper_ml.app_services.reproduction import (
    ReplayExecutionRequest,
    ReproductionService,
)
from temper_ml.app_services.retention import CleanupPlan, RetentionService
from temper_ml.app_services.runs import (
    RunExecutionResult,
    RunLaunchRequest,
    RunService,
)
from temper_ml.domain.artifacts import Artifact
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import (
    CompatibilityGroup,
    RuntimeTargetConstraint,
)
from temper_ml.domain.datasets import (
    DatasetVersion,
    DeduplicationRule,
    FieldMapping,
    FilterRule,
    RendererSpec,
    SplitPart,
    SplitRule,
    renderer_identity,
)
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
    UserDecisionStatus,
)
from temper_ml.domain.experiments import (
    Experiment,
    ExperimentDerivation,
    ReproductionMode,
)
from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.local_use import AdapterExport, LocalUseSession
from temper_ml.domain.policies import BaselinePolicy, PerModelBaseline
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import (
    TypedRecord,
    record_reference,
    thaw_json,
)
from temper_ml.domain.runs import EvaluationMode, Run
from temper_ml.domain.retention import CleanupReceipt
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.runtime.fixture_inference import InferenceSettings
from temper_ml.runtime.preflight import (
    EstimateComponents,
    PreflightEstimate,
    capture_capability_profile,
    estimate_resources,
    preflight,
)
from temper_ml.runtime.recipe_resolution import (
    RecipeCatalogEntry,
    RecipeResolver,
    resolution_view,
)
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore


FIXTURE_PROJECT_ID = "project-fixture-runtime"
FIXTURE_DATASET_ID = "dataset-fixture-runtime"
FIXTURE_CONFIRMATION_SUITE_ID = "suite-fixture-confirmation"
FIXTURE_DEVELOPMENT_SUITE_ID = "suite-fixture-development"
FIXTURE_POLICY_ID = "policy-fixture-recommendation"
FIXTURE_RECOMMENDATION_ID = "recommendation-fixture"


class FixtureTokenizer:
    """Small deterministic tokenizer shared by the CLI and loopback UI."""

    identity = ContentIdentity(
        "sha256", hashlib.sha256(b"temper-public-fixture-tokenizer-v1").hexdigest()
    )

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.encode("utf-8"))


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    label: str
    recipe: Recipe
    resolution: RecipeResolution
    experiment_id: str
    run_id: str
    request_id: str
    artifact_id: str
    estimate: PreflightEstimate

    def to_view(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "resolution": resolution_view(self.resolution),
            "estimate": self.estimate.to_dict(),
        }


@dataclass(frozen=True)
class ComparisonState:
    prompt: Mapping[str, Any]
    settings: InferenceSettings
    outputs: Mapping[str, Mapping[str, Any]]
    artifacts: Mapping[str, Artifact]


@dataclass
class _JourneyState:
    opened: OpenedProject | None = None
    model: BaseModelRevision | None = None
    prepared: PreparedDataset | None = None
    requirements: HardwareRequirements | None = None
    target: ExecutionTarget | None = None
    group: CompatibilityGroup | None = None
    profile: HardwareCapabilityProfile | None = None
    candidates: tuple[CandidateSpec, ...] = ()


@dataclass(frozen=True)
class _FixtureReplayDraft:
    plan: ReplayPlan
    launch: RunLaunchRequest
    candidate_key: str

    def to_view(self) -> dict[str, object]:
        return {**self.plan.to_view(), "candidate_key": self.candidate_key}


class FixtureJourneyService:
    """Staged, idempotent orchestration over existing domain services."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)
        self.state = _JourneyState()
        self._comparison: ComparisonState | None = None
        self._blind_packet: BlindReviewPacket | None = None
        self._sealed_blind_review: Review | None = None
        self._cleanup_plan: CleanupPlan | None = None
        self._replay_draft: _FixtureReplayDraft | None = None

    def setup_project(self) -> dict[str, object]:
        task, model, project, baseline, policy = _fixture_project_records()
        opened = ProjectService(self.project_root).create(
            ProjectCreateRequest(task, project, baseline, policy, (model,))
        )
        self.state.opened = opened
        self.state.model = model
        return opened.to_view()

    def import_dataset(
        self,
        *,
        source_format: str = "fixture",
        source_text: str | None = None,
    ) -> dict[str, object]:
        self._ensure_project()
        request = _dataset_request()
        service = DatasetService(self.project_root)
        if source_format == "fixture":
            source = dumps_canonical_json(_fixture_source_rows())
            prepared = service.import_json(source, request)
        else:
            if not isinstance(source_text, str) or not source_text.strip():
                raise ApplicationServiceError("dataset_source_required")
            source = source_text.encode("utf-8")
            if source_format == "json":
                prepared = service.import_json(source, request)
            elif source_format == "jsonl":
                prepared = service.import_jsonl(source, request)
            elif source_format == "csv":
                prepared = service.import_csv(source, request)
            else:
                raise ApplicationServiceError("dataset_format_unsupported")
        self.state.prepared = prepared
        return {
            "dataset_version": record_reference(prepared.version).to_dict(),
            "statistics": prepared.version.statistics.to_dict(),
            "previews": [preview.to_dict() for preview in prepared.previews],
            "private_preview": True,
        }

    def resolve_candidates(self) -> dict[str, object]:
        opened, model, prepared = self._prepared_context()
        del opened
        requirements, target = _fixture_hardware()
        entries = _fixture_recipe_entries()
        resolver = RecipeResolver()
        resolutions = tuple(
            resolver.resolve(
                entry,
                base_model_revision=model,
                hardware_requirements=requirements,
                execution_target=target,
            )
            for entry in entries
        )
        group = CompatibilityGroup(
            group_id="group-fixture-runtime",
            base_model_revision=record_reference(model),
            tokenizer_identity=model.tokenizer_identity,
            rendering_template=_fixture_rendering_contract(),
            adapter_type=resolutions[0].adapter_type,
            target_modules=resolutions[0].target_modules,
            runtime_targets=(
                RuntimeTargetConstraint(
                    target.target_class,
                    target.accelerator_backend,
                    target.runtime_contract,
                    ("fixture_adapter",),
                ),
            ),
            merge_methods=(),
        )
        profile = capture_capability_profile(
            profile_id="profile-fixture-runtime",
            execution_target=target,
            accelerator_backend="none",
            accelerator_architecture="fixture-cpu",
            accelerator_model="Synthetic fixture CPU",
            accelerator_count=0,
            accelerator_memory_bytes=(),
            system_memory_bytes=1_000_000,
            supported_precision_modes=("fp32",),
            supported_quantization_modes=(),
            capabilities=("fixture_adapter",),
            library_versions={"fixture_runtime": "v1"},
        )
        estimates = tuple(
            estimate_resources(
                resolution,
                EstimateComponents(
                    base_model_bytes=0,
                    adapter_optimizer_bytes=0,
                    peak_activation_bytes=0,
                    accelerator_runtime_overhead_bytes=0,
                    dataset_bytes=len(prepared.rendered_bytes),
                    host_runtime_overhead_bytes=1024,
                ),
            )
            for resolution in resolutions
        )
        candidates = (
            CandidateSpec(
                "ember",
                "Ember / balanced",
                entries[0].recipe,
                resolutions[0],
                "experiment-fixture-runtime",
                "run-fixture-runtime",
                "request-fixture-runtime",
                "artifact-fixture-runtime",
                estimates[0],
            ),
            CandidateSpec(
                "slate",
                "Slate / capacity",
                entries[1].recipe,
                resolutions[1],
                "experiment-fixture-challenger",
                "run-fixture-challenger",
                "request-fixture-challenger",
                "artifact-fixture-challenger",
                estimates[1],
            ),
        )
        self.state.requirements = requirements
        self.state.target = target
        self.state.group = group
        self.state.profile = profile
        self.state.candidates = candidates
        return {
            "execution_target": record_reference(target).to_dict(),
            "offline": True,
            "candidates": [
                {
                    **candidate.to_view(),
                    "preflight": preflight(
                        candidate.resolution,
                        requirements,
                        target,
                        profile,
                        candidate.estimate,
                    ).to_view(),
                }
                for candidate in candidates
            ],
        }

    def launch_candidates(self) -> dict[str, object]:
        if not self.state.candidates:
            self.resolve_candidates()
        results = tuple(self._launch(candidate) for candidate in self.state.candidates)
        return {"runs": [result.to_view() for result in results]}

    def launch_primary(self) -> RunExecutionResult:
        if not self.state.candidates:
            self.resolve_candidates()
        return self._launch(self.state.candidates[0])

    def _launch(self, candidate: CandidateSpec) -> RunExecutionResult:
        opened, model, prepared = self._prepared_context()
        requirements, target, group, profile = self._resolved_context()
        experiment = ExperimentService(self.project_root).freeze(
            ExperimentFreezeRequest(
                experiment_id=candidate.experiment_id,
                opened_project=opened,
                dataset_version=prepared.version.identity,
                base_model_revision=model,
                recipe=candidate.recipe,
                recipe_resolution=candidate.resolution,
                compatibility_group=group,
                hardware_requirements=requirements,
                execution_target=target,
            )
        )
        request = RunLaunchRequest(
            run_id=candidate.run_id,
            request_id=candidate.request_id,
            artifact_id=candidate.artifact_id,
            experiment=experiment,
            recipe_resolution=candidate.resolution,
            prepared_dataset=prepared,
            base_model_revision=model,
            compatibility_group=group,
            hardware_requirements=requirements,
            execution_target=target,
            hardware_capability_profile=profile,
            estimate=candidate.estimate,
            evaluation_mode=EvaluationMode.NO_QUALITY_EVALUATION,
        )
        service = RunService(self.project_root)
        try:
            return service.reopen_completed(request)
        except ApplicationServiceError as exc:
            if exc.code != "run_not_found":
                raise
        return service.launch(request)

    def compare(
        self,
        *,
        prompt: str,
        maximum_tokens: int = 64,
        seed: int = 17,
    ) -> dict[str, object]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApplicationServiceError("playground_prompt_required")
        settings = InferenceSettings(
            temperature=0,
            maximum_tokens=maximum_tokens,
            seed=seed,
        )
        artifacts = self._candidate_artifacts()
        outputs: dict[str, Mapping[str, Any]] = {}
        views: list[dict[str, object]] = []
        local = LocalUseService(self.project_root)
        for key in ("ember", "slate"):
            artifact = artifacts[key]
            model, group, target = self._artifact_dependencies(artifact)
            result = local.focused(
                LocalUseRequest(
                    artifact=artifact,
                    base_model_revision=model,
                    compatibility_group=group,
                    execution_target=target,
                    settings=settings,
                    inputs=({"text": prompt},),
                )
            )
            output = thaw_json(result.inference.outputs[0])
            if not isinstance(output, dict):
                raise ApplicationServiceError("playground_output_invalid")
            outputs[key] = output
            views.append(
                {
                    "key": key,
                    "label": _candidate_label(key),
                    "artifact": record_reference(artifact).to_dict(),
                    "output": output,
                    "integrity": result.integrity.to_receipt(),
                }
            )
        self._comparison = ComparisonState(
            {"text": prompt}, settings, outputs, artifacts
        )
        return {
            "prompt": {"text": prompt},
            "settings": settings.to_dict(),
            "outputs": views,
            "synchronized": True,
            "saved": False,
        }

    def record_solo_review(
        self,
        *,
        notes: str,
        ratings: Mapping[str, int],
        declaration: str,
    ) -> dict[str, object]:
        comparison = self._require_comparison()
        if not isinstance(notes, str) or not notes.strip():
            raise ApplicationServiceError("review_notes_required")
        if not isinstance(declaration, str) or not declaration.strip():
            raise ApplicationServiceError("review_declaration_required")
        aliases = ("ember", "slate")
        if set(ratings) != set(aliases):
            raise ApplicationServiceError("review_ratings_invalid")
        review_id = self._next_logical_id("review-fixture-solo", Review)
        review = Review(
            review_id=review_id,
            mode=ReviewMode.SOLO,
            stage=ReviewStage.RECORDED,
            entries=(
                ReviewEntry(
                    prompt_id=f"prompt-{review_id}",
                    prompt=comparison.prompt,
                    settings=comparison.settings.to_dict(),
                    outputs=tuple(
                        ReviewOutput(alias, comparison.outputs[alias])
                        for alias in aliases
                    ),
                    notes=notes,
                    ratings=tuple(
                        ReviewRating(alias, "task_fit", ratings[alias])
                        for alias in aliases
                    ),
                ),
            ),
            reviewer_declaration=declaration,
            candidate_mappings=tuple(
                ReviewCandidate(alias, record_reference(comparison.artifacts[alias]))
                for alias in aliases
            ),
            leak_audit_passed=False,
        )
        recorded = EvaluationService(self.project_root).record_solo_review(review)
        return {
            "review": record_reference(recorded).to_dict(),
            "mode": recorded.mode.value,
            "stage": recorded.stage.value,
            "structured_review": True,
        }

    def prepare_blind_review(self) -> dict[str, object]:
        comparison = self._require_comparison()
        packet_id = self._next_logical_id("review-fixture-blind", Review)
        packet = EvaluationService(self.project_root).prepare_blind_review(
            packet_id,
            (
                BlindReviewInput(
                    prompt_id=f"prompt-{packet_id}",
                    prompt=comparison.prompt,
                    settings=comparison.settings.to_dict(),
                    outputs=tuple(
                        BlindCandidateOutput(
                            record_reference(comparison.artifacts[key]),
                            comparison.outputs[key],
                        )
                        for key in ("ember", "slate")
                    ),
                ),
            ),
        )
        self._blind_packet = packet
        self._sealed_blind_review = None
        return {
            "packet": packet.public_fields(),
            "leak_audit_passed": packet.leak_audit_passed,
            "identities_revealed": False,
        }

    def seal_blind_review(
        self,
        *,
        notes: str,
        ratings: Mapping[str, int],
        declaration: str,
    ) -> dict[str, object]:
        packet = self._blind_packet
        if packet is None:
            raise ApplicationServiceError("blind_review_prepare_required")
        aliases = packet.aliases
        if set(ratings) != set(aliases):
            raise ApplicationServiceError("blind_review_ratings_invalid")
        judgment = BlindReviewJudgment(
            packet.entries[0].prompt_id,
            notes,
            tuple(ReviewRating(alias, "task_fit", ratings[alias]) for alias in aliases),
        )
        sealed = EvaluationService(self.project_root).seal_blind_review(
            packet.packet_id,
            packet,
            (judgment,),
            reviewer_declaration=declaration,
        )
        self._sealed_blind_review = sealed
        return {
            "review": record_reference(sealed).to_dict(),
            "stage": sealed.stage.value,
            "judgment_sealed": True,
            "identities_revealed": False,
        }

    def reveal_blind_review(self) -> dict[str, object]:
        if self._blind_packet is None or self._sealed_blind_review is None:
            raise ApplicationServiceError("blind_review_seal_required")
        revealed = EvaluationService(self.project_root).reveal_blind_review(
            self._sealed_blind_review,
            self._blind_packet,
        )
        return {
            "review": record_reference(revealed).to_dict(),
            "stage": revealed.stage.value,
            "identities_revealed": True,
            "candidate_mappings": [
                mapping.to_dict() for mapping in revealed.candidate_mappings
            ],
        }

    def evaluate_candidates(self) -> dict[str, object]:
        artifacts = self._candidate_artifacts()
        current_integrity = {}
        local_use = LocalUseService(self.project_root)
        for key in ("ember", "slate"):
            artifact = artifacts[key]
            model, group, target = self._artifact_dependencies(artifact)
            current_integrity[key] = local_use.inspect_artifact(
                artifact,
                model,
                group,
                target,
            ).integrity
        service = EvaluationService(self.project_root)
        development = self._ensure_suite(service, CaseSuiteKind.DEVELOPMENT)
        confirmation = self._ensure_suite(service, CaseSuiteKind.CONFIRMATION)
        if confirmation.state is SuiteEvidenceState.UNSEALED:
            confirmation = service.seal_suite(confirmation)
        policy = service.register_policy(_fixture_recommendation_policy())
        results: list[EvaluationResult] = []
        for key in ("ember", "slate"):
            artifact = artifacts[key]
            result = EvaluationResult(
                result_id=f"result-fixture-{key}",
                candidate=record_reference(artifact),
                evaluation_mode=EvaluationMode.FULL_SUITE,
                artifact_integrity_status=ArtifactIntegrityStatus.PASSED,
                artifact_integrity_evidence=current_integrity[key].evidence_identity,
                evidence_status=EvidenceStatus.PASSED,
                metrics=(
                    MetricObservation(
                        "fixture_task_fit",
                        EvaluatorKind.TASK_METRIC,
                        1,
                        MetricDirection.MAXIMIZE,
                    ),
                    MetricObservation(
                        "format_validity",
                        EvaluatorKind.FORMAT_CHECK,
                        1,
                        MetricDirection.MAXIMIZE,
                    ),
                ),
                suite=record_reference(confirmation),
                suite_state=confirmation.state,
                conflicts=("fixture_candidates_tied",),
            )
            results.append(service.record_result(result))
        recommendation = service.recommend(
            FIXTURE_RECOMMENDATION_ID, policy, tuple(results)
        )
        return {
            "development_suite": record_reference(development).to_dict(),
            "confirmation_suite": record_reference(confirmation).to_dict(),
            "results": [record_reference(result).to_dict() for result in results],
            "recommendation": recommendation.to_payload(),
            "recommendation_reference": record_reference(recommendation).to_dict(),
            "synthetic_fixture_evidence": True,
        }

    def capture_review(
        self,
        review_identity: ContentIdentity,
        *,
        suite_kind: str = "development",
    ) -> dict[str, object]:
        try:
            kind = CaseSuiteKind(suite_kind)
        except ValueError:
            raise ApplicationServiceError("evaluation_suite_kind_invalid") from None
        if kind not in (CaseSuiteKind.DEVELOPMENT, CaseSuiteKind.REGRESSION):
            raise ApplicationServiceError("evaluation_capture_suite_invalid")
        if not isinstance(review_identity, ContentIdentity):
            raise ApplicationServiceError("evaluation_capture_review_required")
        reviews = [
            review
            for review in self._records(Review)
            if review.identity == review_identity
        ]
        if len(reviews) != 1:
            raise ApplicationServiceError("evaluation_capture_review_not_found")
        review = reviews[0]
        if review.stage not in {
            ReviewStage.RECORDED,
            ReviewStage.BLIND_SEALED,
            ReviewStage.BLIND_REVEALED,
        }:
            raise ApplicationServiceError("evaluation_capture_review_incomplete")
        service = EvaluationService(self.project_root)
        suite = self._ensure_suite(service, kind)
        case_id = self._next_case_id(suite, kind)
        revised = service.modify_suite(
            suite,
            cases=(*suite.cases, EvaluationCase(case_id, review.identity)),
        )
        return {
            "case_id": case_id,
            "case_content_identity": {
                "algorithm": review.identity.algorithm,
                "value": review.identity.value,
            },
            "suite": record_reference(revised).to_dict(),
            "suite_state": revised.state.value,
        }

    def record_decision(
        self,
        *,
        candidate_key: str,
        status: str = "selected",
        override_reason: str | None = None,
    ) -> dict[str, object]:
        recommendation = self._latest_recommendation()
        artifacts = self._candidate_artifacts()
        if candidate_key not in artifacts:
            raise ApplicationServiceError("candidate_not_found")
        try:
            decision_status = UserDecisionStatus(status)
        except ValueError:
            raise ApplicationServiceError("user_decision_status_invalid") from None
        if decision_status not in (
            UserDecisionStatus.SELECTED,
            UserDecisionStatus.PINNED,
            UserDecisionStatus.REJECTED,
        ):
            raise ApplicationServiceError("user_decision_status_unsupported")
        candidate = record_reference(artifacts[candidate_key])
        assessment = next(
            (
                item
                for item in recommendation.assessments
                if item.candidate == candidate
            ),
            None,
        )
        if assessment is None:
            raise ApplicationServiceError("user_decision_candidate_unassessed")
        decision = UserDecision(
            decision_id=self._next_logical_id("decision-fixture", UserDecision),
            recommendation=record_reference(recommendation),
            candidate=candidate,
            status=decision_status,
            evidence_status_at_decision=assessment.evidence_status,
            override_reason=override_reason,
        )
        recorded = EvaluationService(self.project_root).record_decision(decision)
        return {
            "decision": record_reference(recorded).to_dict(),
            "status": recorded.status.value,
            "candidate_key": candidate_key,
            "evidence_status": recorded.evidence_status_at_decision.value,
            "recommendation_unchanged": True,
        }

    def focused_local_use(
        self,
        *,
        candidate_key: str,
        prompt: str,
        maximum_tokens: int = 64,
        seed: int = 17,
        save: bool = True,
        capture_identity: ContentIdentity | None = None,
    ) -> dict[str, object]:
        artifact = self._authorized_artifact(candidate_key)
        model, group, target = self._artifact_dependencies(artifact)
        session_id = (
            self._next_logical_id(f"session-fixture-{candidate_key}", LocalUseSession)
            if save
            else None
        )
        captures = (capture_identity,) if capture_identity is not None else ()
        result = LocalUseService(self.project_root).focused(
            LocalUseRequest(
                artifact=artifact,
                base_model_revision=model,
                compatibility_group=group,
                execution_target=target,
                settings=InferenceSettings(0, maximum_tokens, seed),
                inputs=({"text": prompt},),
                session_id=session_id,
                evaluation_captures=captures,
            )
        )
        view = result.to_view()
        view["candidate_key"] = candidate_key
        view["focused_local_use"] = True
        view["general_chat"] = False
        return view

    def batch_local_use(
        self,
        *,
        candidate_key: str,
        prompts: tuple[str, ...],
        maximum_tokens: int = 64,
        seed: int = 17,
        save: bool = False,
    ) -> dict[str, object]:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ApplicationServiceError("local_batch_inputs_invalid")
        artifact = self._authorized_artifact(candidate_key)
        model, group, target = self._artifact_dependencies(artifact)
        session_id = (
            self._next_logical_id(
                f"session-fixture-batch-{candidate_key}", LocalUseSession
            )
            if save
            else None
        )
        result = LocalUseService(self.project_root).batch(
            LocalUseRequest(
                artifact=artifact,
                base_model_revision=model,
                compatibility_group=group,
                execution_target=target,
                settings=InferenceSettings(0, maximum_tokens, seed),
                inputs=tuple({"text": prompt} for prompt in prompts),
                session_id=session_id,
            )
        )
        view = result.to_view()
        view["candidate_key"] = candidate_key
        view["batch_size"] = len(prompts)
        view["general_chat"] = False
        return view

    def export_selected(self, *, candidate_key: str) -> dict[str, object]:
        artifact = self._authorized_artifact(candidate_key)
        model, group, target = self._artifact_dependencies(artifact)
        export_id = f"export-fixture-{candidate_key}"
        exported = LocalUseService(self.project_root).export(
            AdapterExportRequest(
                export_id,
                artifact,
                model,
                group,
                target,
            )
        )
        return exported.to_view()

    def legacy_workflow(self, evaluation_mode: EvaluationMode) -> dict[str, object]:
        if evaluation_mode is not EvaluationMode.NO_QUALITY_EVALUATION:
            raise ApplicationServiceError("run_evaluation_mode_not_supported")
        self.setup_project()
        self.import_dataset()
        self.resolve_candidates()
        run = self.launch_primary()
        if run.artifact is None:
            raise ApplicationServiceError("fixture_workflow_artifact_missing")
        opened, model, prepared = self._prepared_context()
        del opened
        _, target, group, _ = self._resolved_context()
        local = LocalUseService(self.project_root)
        focused = local.focused(
            LocalUseRequest(
                artifact=run.artifact,
                base_model_revision=model,
                compatibility_group=group,
                execution_target=target,
                settings=InferenceSettings(0, 32, 17),
                inputs=({"text": "Synthetic focused prompt"},),
                session_id="session-fixture-focused",
            )
        )
        batch = local.batch(
            LocalUseRequest(
                artifact=run.artifact,
                base_model_revision=model,
                compatibility_group=group,
                execution_target=target,
                settings=InferenceSettings(0, 32, 17),
                inputs=(
                    {"text": "Synthetic batch prompt one"},
                    {"text": "Synthetic batch prompt two"},
                ),
            )
        )
        exported = local.export(
            AdapterExportRequest(
                "export-fixture-runtime", run.artifact, model, group, target
            )
        )
        local.verify_export(
            exported.record,
            artifact=run.artifact,
            base_model_revision=model,
            compatibility_group=group,
            execution_target=target,
        )
        store = TypedEvidenceStore(self.project_root)
        verification = store.verify()
        public = store.public_dump().value
        project = self.state.opened.project if self.state.opened is not None else None
        if project is None:
            raise ApplicationServiceError("fixture_project_missing")
        candidate = self.state.candidates[0]
        experiment = self._record_by_logical_id(Experiment, candidate.experiment_id)
        return {
            "schema_version": "v1",
            "command": "fixture-workflow",
            "status": "verified",
            "evaluation_mode": evaluation_mode.value,
            "project": record_reference(project).to_dict(),
            "dataset_version": record_reference(prepared.version).to_dict(),
            "recipe_resolution": record_reference(candidate.resolution).to_dict(),
            "experiment": record_reference(experiment).to_dict(),
            "run": record_reference(run.run).to_dict(),
            "artifact": record_reference(run.artifact).to_dict(),
            "focused_session": (
                record_reference(focused.session).to_dict()
                if focused.session is not None
                else None
            ),
            "batch_output_count": len(batch.inference.outputs),
            "adapter_export": record_reference(exported.record).to_dict(),
            "store": verification.to_dict(),
            "public_projection_verified": (
                public["classification"] == "public_projection"
            ),
            "hosted_deployment": False,
            "deployment_ready": False,
        }

    def workspace(self) -> dict[str, object]:
        ReproductionService(self.project_root).reconcile_pending()
        previews = (
            [preview.to_dict() for preview in self.state.prepared.previews]
            if self.state.prepared is not None
            else []
        )
        return WorkspaceQueryService(self.project_root).view(
            prepared_available=self.state.prepared is not None,
            private_previews=previews,
            active_cleanup_plan=(
                self._cleanup_plan.to_view() if self._cleanup_plan is not None else None
            ),
            active_replay_plan=(
                self._replay_draft.to_view() if self._replay_draft is not None else None
            ),
        )

    def storage_inventory(self) -> dict[str, object]:
        """Return the complete fixed-root inventory without implicit selection."""

        return RetentionService(self.project_root).inventory().to_view()

    def preview_cleanup(self, entry_ids: tuple[str, ...]) -> dict[str, object]:
        """Cache one exact cleanup plan for a later explicit confirmation."""

        plan = RetentionService(self.project_root).plan(entry_ids)
        self._cleanup_plan = plan
        return plan.to_view()

    def execute_cleanup(
        self,
        plan_id: str,
        *,
        confirm: bool,
        entry_ids: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """Execute only the currently displayed, exactly identified cleanup plan."""

        plan = self._cleanup_plan
        if plan is None:
            raise ApplicationServiceError("cleanup_plan_required")
        if not isinstance(plan_id, str) or plan_id != plan.plan_id:
            raise ApplicationServiceError("cleanup_plan_mismatch")
        if entry_ids is not None and (
            not isinstance(entry_ids, tuple)
            or tuple(sorted(entry_ids)) != plan.selected_entry_ids
        ):
            raise ApplicationServiceError("cleanup_selection_plan_mismatch")
        receipt = RetentionService(self.project_root).execute(plan, confirm=confirm)
        self._cleanup_plan = None
        return _cleanup_receipt_view(receipt)

    def prepare_replay(self, candidate_key: str, mode: str) -> dict[str, object]:
        """Prepare a strict replay or a visibly derived adapted reproduction."""

        ReproductionService(self.project_root).reconcile_pending()
        if self.state.prepared is None:
            self.import_dataset()
        if not self.state.candidates:
            self.resolve_candidates()
        candidate = next(
            (item for item in self.state.candidates if item.key == candidate_key),
            None,
        )
        if candidate is None:
            raise ApplicationServiceError("candidate_not_found")
        _, model, prepared = self._prepared_context()
        requirements, target, group, profile = self._resolved_context()
        source = self._record_by_logical_id(Experiment, candidate.experiment_id)
        ordinal = (
            sum(
                run.run_id.startswith(f"run-replay-{candidate.key}-")
                for run in self._records(Run)
            )
            + 1
        )
        suffix = f"{candidate.key}-{ordinal:03d}"
        if mode == ReplayMode.STRICT.value:
            exact_preflight = preflight(
                candidate.resolution,
                requirements,
                target,
                profile,
                candidate.estimate,
            )
            plan = strict_replay_plan(source, exact_preflight)
            launch = RunLaunchRequest(
                run_id=f"run-replay-{suffix}",
                request_id=f"request-replay-{suffix}",
                artifact_id=f"artifact-replay-{suffix}",
                experiment=source,
                recipe_resolution=candidate.resolution,
                prepared_dataset=prepared,
                base_model_revision=model,
                compatibility_group=group,
                hardware_requirements=requirements,
                execution_target=target,
                hardware_capability_profile=profile,
                estimate=candidate.estimate,
            )
        elif mode == ReplayMode.ADAPTED.value:
            derivation_ordinal = len(self._records(ExperimentDerivation)) + 1
            adapted_requirements = replace(
                requirements,
                requirements_id=f"requirements-replay-{suffix}",
                required_precision_modes=("bf16",),
            )
            adapted_resolution = replace(
                candidate.resolution,
                resolution_id=f"resolution-replay-{suffix}",
                hardware_requirements=record_reference(adapted_requirements),
                precision="bf16",
                applied_constraints=tuple(
                    sorted(
                        {
                            *candidate.resolution.applied_constraints,
                            "adapted_precision_bf16",
                        }
                    )
                ),
            )
            adapted_profile = replace(
                profile,
                profile_id=f"profile-replay-{suffix}",
                supported_precision_modes=("bf16",),
            )
            strict_preflight = preflight(
                candidate.resolution,
                requirements,
                target,
                adapted_profile,
                candidate.estimate,
            )
            adapted_preflight = preflight(
                adapted_resolution,
                adapted_requirements,
                target,
                adapted_profile,
                candidate.estimate,
            )
            derivation = ExperimentService(self.project_root).clone(
                source,
                experiment_id=f"experiment-replay-{suffix}",
                replacements={
                    "recipe_resolution": record_reference(adapted_resolution),
                    "hardware_requirements": record_reference(adapted_requirements),
                },
                derivation_id=f"derivation-adapted-{derivation_ordinal:03d}",
                diff_id=f"manifest-diff-adapted-{derivation_ordinal:03d}",
                reason_code="hardware_precision_adaptation",
                reason=(
                    "The available fixture profile supports bf16 instead of the "
                    "source experiment's fp32 requirement."
                ),
                reproduction_mode=ReproductionMode.ADAPTED_REPRODUCTION,
                supporting_records=(adapted_requirements, adapted_resolution),
            )
            plan = plan_replay(
                source,
                strict_preflight,
                adapted_derivation=derivation,
                adapted_preflight=adapted_preflight,
            )
            launch = RunLaunchRequest(
                run_id=f"run-replay-{suffix}",
                request_id=f"request-replay-{suffix}",
                artifact_id=f"artifact-replay-{suffix}",
                experiment=derivation.derived_experiment,
                recipe_resolution=adapted_resolution,
                prepared_dataset=prepared,
                base_model_revision=model,
                compatibility_group=group,
                hardware_requirements=adapted_requirements,
                execution_target=target,
                hardware_capability_profile=adapted_profile,
                estimate=candidate.estimate,
            )
        else:
            raise ApplicationServiceError("replay_mode_invalid")
        self._replay_draft = _FixtureReplayDraft(plan, launch, candidate_key)
        return self._replay_draft.to_view()

    def execute_replay(
        self,
        plan_id: str,
        *,
        candidate_key: str | None = None,
        mode: str | None = None,
    ) -> dict[str, object]:
        """Execute only the currently displayed ready replay plan as a new run."""

        draft = self._replay_draft
        if draft is None:
            return (
                ReproductionService(self.project_root)
                .reconcile_plan(
                    plan_id,
                    candidate_key=candidate_key,
                    mode=mode,
                )
                .to_view()
            )
        if not isinstance(plan_id, str) or plan_id != draft.plan.plan_id:
            raise ApplicationServiceError("replay_plan_mismatch")
        if candidate_key is not None and candidate_key != draft.candidate_key:
            raise ApplicationServiceError("replay_candidate_plan_mismatch")
        if mode is not None and mode != draft.plan.mode.value:
            raise ApplicationServiceError("replay_mode_plan_mismatch")
        result = ReproductionService(self.project_root).execute(
            ReplayExecutionRequest(draft.plan, draft.launch, draft.candidate_key)
        )
        self._replay_draft = None
        return result.to_view()

    def _ensure_project(self) -> None:
        if self.state.opened is None or self.state.model is None:
            self.setup_project()

    def _prepared_context(
        self,
    ) -> tuple[OpenedProject, BaseModelRevision, PreparedDataset]:
        self._ensure_project()
        if self.state.prepared is None:
            if self._records(DatasetVersion):
                raise ApplicationServiceError("dataset_reimport_required")
            raise ApplicationServiceError("dataset_import_required")
        if self.state.opened is None or self.state.model is None:
            raise ApplicationServiceError("fixture_project_missing")
        return self.state.opened, self.state.model, self.state.prepared

    def _resolved_context(
        self,
    ) -> tuple[
        HardwareRequirements,
        ExecutionTarget,
        CompatibilityGroup,
        HardwareCapabilityProfile,
    ]:
        values = (
            self.state.requirements,
            self.state.target,
            self.state.group,
            self.state.profile,
        )
        if any(value is None for value in values):
            raise ApplicationServiceError("candidate_resolution_required")
        requirements, target, group, profile = values
        if not isinstance(requirements, HardwareRequirements):
            raise ApplicationServiceError("candidate_resolution_invalid")
        if not isinstance(target, ExecutionTarget):
            raise ApplicationServiceError("candidate_resolution_invalid")
        if not isinstance(group, CompatibilityGroup):
            raise ApplicationServiceError("candidate_resolution_invalid")
        if not isinstance(profile, HardwareCapabilityProfile):
            raise ApplicationServiceError("candidate_resolution_invalid")
        return requirements, target, group, profile

    def _candidate_artifacts(self) -> dict[str, Artifact]:
        artifacts = {
            artifact.artifact_id: artifact for artifact in self._records(Artifact)
        }
        expected = {
            "ember": "artifact-fixture-runtime",
            "slate": "artifact-fixture-challenger",
        }
        if any(artifact_id not in artifacts for artifact_id in expected.values()):
            raise ApplicationServiceError("candidate_runs_required")
        return {key: artifacts[artifact_id] for key, artifact_id in expected.items()}

    def _artifact_dependencies(
        self, artifact: Artifact
    ) -> tuple[BaseModelRevision, CompatibilityGroup, ExecutionTarget]:
        store = TypedEvidenceStore(self.project_root)
        try:
            model = store.read_record(artifact.base_model_revision).record
            if len(artifact.compatibility_groups) != 1:
                raise ApplicationServiceError("local_use_compatibility_ambiguous")
            group = store.read_record(artifact.compatibility_groups[0]).record
            run = store.read_record(artifact.producing_run).record
            if not isinstance(run, Run):
                raise ApplicationServiceError("local_use_dependency_invalid")
            target = store.read_record(run.execution_target).record
        except EvidenceError:
            raise ApplicationServiceError("local_use_dependency_missing") from None
        if not isinstance(model, BaseModelRevision):
            raise ApplicationServiceError("local_use_dependency_invalid")
        if not isinstance(group, CompatibilityGroup):
            raise ApplicationServiceError("local_use_dependency_invalid")
        if not isinstance(target, ExecutionTarget):
            raise ApplicationServiceError("local_use_dependency_invalid")
        return model, group, target

    def _authorized_artifact(self, candidate_key: str) -> Artifact:
        artifacts = self._candidate_artifacts()
        if candidate_key not in artifacts:
            raise ApplicationServiceError("candidate_not_found")
        artifact = artifacts[candidate_key]
        current = EvaluationService(self.project_root).current_decision()
        permitted = (
            current is not None
            and current.candidate == record_reference(artifact)
            and current.status
            in (UserDecisionStatus.SELECTED, UserDecisionStatus.PINNED)
        )
        if not permitted:
            raise ApplicationServiceError("local_use_selection_required")
        return artifact

    def _latest_recommendation(self) -> Recommendation:
        recommendations = self._records(Recommendation)
        matches = [
            item
            for item in recommendations
            if item.recommendation_id == FIXTURE_RECOMMENDATION_ID
        ]
        if len(matches) != 1:
            raise ApplicationServiceError("recommendation_required")
        return matches[0]

    def _ensure_suite(
        self, service: EvaluationService, kind: CaseSuiteKind
    ) -> EvaluationSuite:
        suite_id = (
            FIXTURE_CONFIRMATION_SUITE_ID
            if kind is CaseSuiteKind.CONFIRMATION
            else (
                FIXTURE_DEVELOPMENT_SUITE_ID
                if kind is CaseSuiteKind.DEVELOPMENT
                else "suite-fixture-regression"
            )
        )
        existing = [
            suite
            for suite in self._records(EvaluationSuite)
            if suite.suite_id == suite_id
        ]
        if existing:
            superseded = {
                suite.prior_suite.identity
                for suite in existing
                if suite.prior_suite is not None
            }
            current = [suite for suite in existing if suite.identity not in superseded]
            if len(current) != 1:
                raise ApplicationServiceError("evaluation_suite_revision_ambiguous")
            return current[0]
        return service.register_suite(_fixture_suite(suite_id, kind))

    def _require_comparison(self) -> ComparisonState:
        if self._comparison is None:
            raise ApplicationServiceError("playground_comparison_required")
        return self._comparison

    def _records(self, kind: type[Any]) -> tuple[Any, ...]:
        store = TypedEvidenceStore(self.project_root)
        try:
            return tuple(
                stored.record
                for stored in store.iter_records()
                if isinstance(stored.record, kind)
            )
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None

    def _record_by_logical_id(self, kind: type[Any], logical_id: str) -> Any:
        matches = [
            record
            for record in self._records(kind)
            if _logical_id(record) == logical_id
        ]
        if len(matches) != 1:
            raise ApplicationServiceError("fixture_record_missing")
        return matches[0]

    def _next_logical_id(self, prefix: str, kind: type[Any]) -> str:
        count = sum(
            _logical_id(record).startswith(prefix) for record in self._records(kind)
        )
        return f"{prefix}-{count + 1:03d}"

    @staticmethod
    def _next_case_id(suite: EvaluationSuite, kind: CaseSuiteKind) -> str:
        prefix = f"case-captured-{kind.value}"
        count = sum(case.case_id.startswith(prefix) for case in suite.cases)
        return f"{prefix}-{count + 1:03d}"


class WorkspaceQueryService:
    """Verified, read-only aggregate views for the loopback UI."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)

    def view(
        self,
        *,
        prepared_available: bool = False,
        private_previews: list[dict[str, object]] | None = None,
        active_cleanup_plan: dict[str, object] | None = None,
        active_replay_plan: dict[str, object] | None = None,
    ) -> dict[str, object]:
        store = TypedEvidenceStore(self.project_root)
        try:
            store.verify()
            inventory = RetentionService(self.project_root).inventory().to_view()
            checkpoint_identities = _verified_checkpoint_identities(inventory)
            verification = store.verify()
            stored = store.iter_records()
            streams = store.iter_streams()
        except EvidenceError as exc:
            if exc.code in {
                "project_not_found",
                "store_missing",
                "store_not_found",
                "store_root_missing",
            }:
                return _empty_workspace()
            raise ApplicationServiceError(exc.code) from None
        records = tuple(item.record for item in stored)
        projects = tuple(item for item in records if isinstance(item, Project))
        datasets = tuple(item for item in records if isinstance(item, DatasetVersion))
        resolutions = tuple(
            item for item in records if isinstance(item, RecipeResolution)
        )
        runs = tuple(item for item in records if isinstance(item, Run))
        artifacts = tuple(item for item in records if isinstance(item, Artifact))
        suites = tuple(item for item in records if isinstance(item, EvaluationSuite))
        reviews = tuple(item for item in records if isinstance(item, Review))
        results = tuple(item for item in records if isinstance(item, EvaluationResult))
        recommendations = tuple(
            item for item in records if isinstance(item, Recommendation)
        )
        decisions = EvaluationService(self.project_root).decision_history()
        current_decision = decisions[-1] if decisions else None
        sessions = tuple(item for item in records if isinstance(item, LocalUseSession))
        exports = tuple(item for item in records if isinstance(item, AdapterExport))
        cleanup_receipts = tuple(
            item for item in records if isinstance(item, CleanupReceipt)
        )
        derivations = tuple(
            item for item in records if isinstance(item, ExperimentDerivation)
        )
        run_streams = {snapshot.stream_id: snapshot.events for snapshot in streams}
        candidate_key_by_id = {
            "artifact-fixture-runtime": "ember",
            "artifact-fixture-challenger": "slate",
        }
        return {
            "schema_version": "v1",
            "status": "verified",
            "fixture_mode": True,
            "offline": True,
            "general_chat": False,
            "external_dashboard": False,
            "hosted_deployment": False,
            "store": verification.to_dict(),
            "project": (
                {
                    "reference": record_reference(projects[0]).to_dict(),
                    "display_name": projects[0].display_name,
                    "purpose": projects[0].purpose,
                }
                if projects
                else None
            ),
            "dataset": (
                {
                    "reference": record_reference(datasets[-1]).to_dict(),
                    "statistics": datasets[-1].statistics.to_dict(),
                    "rendered_bytes_count": datasets[-1].rendered_bytes_count,
                    "previews": private_previews or [],
                    "prepared_bytes_available": prepared_available,
                    "reimport_required": bool(datasets)
                    and not prepared_available
                    and not artifacts,
                }
                if datasets
                else None
            ),
            "resolutions": [
                {
                    "reference": record_reference(resolution).to_dict(),
                    "rank": resolution.rank,
                    "alpha": resolution.alpha,
                    "seed": resolution.seed,
                    "training_steps": resolution.training_steps,
                    "target_modules": list(resolution.target_modules),
                }
                for resolution in resolutions
            ],
            "runs": [
                _run_view(
                    run,
                    run_streams.get(f"run-{run.run_id}", ()),
                    checkpoint_identities.get(run.run_id, frozenset()),
                )
                for run in runs
            ],
            "artifacts": [
                self._artifact_view(
                    store,
                    artifact,
                    candidate_key_by_id.get(artifact.artifact_id),
                )
                for artifact in artifacts
            ],
            "evaluation": {
                "suites": [
                    {
                        "reference": record_reference(suite).to_dict(),
                        "kind": suite.kind.value,
                        "state": suite.state.value,
                        "case_count": len(suite.cases),
                    }
                    for suite in _current_suites(suites)
                ],
                "reviews": [
                    {
                        "reference": record_reference(review).to_dict(),
                        "mode": review.mode.value,
                        "stage": review.stage.value,
                        "prompt_count": len(review.entries),
                        "identities_revealed": (
                            review.stage is ReviewStage.BLIND_REVEALED
                        ),
                    }
                    for review in _current_reviews(reviews)
                ],
                "results": [
                    {
                        "reference": record_reference(result).to_dict(),
                        "candidate": result.candidate.to_dict(),
                        "evidence_status": result.evidence_status.value,
                        "suite_state": (
                            result.suite_state.value if result.suite_state else None
                        ),
                        "conflicts": list(result.conflicts),
                    }
                    for result in results
                ],
            },
            "recommendation": (
                recommendations[-1].to_payload() if recommendations else None
            ),
            "registry": [
                {
                    "reference": record_reference(decision).to_dict(),
                    "candidate": decision.candidate.to_dict(),
                    "status": decision.status.value,
                    "evidence_status": decision.evidence_status_at_decision.value,
                    "override_reason": decision.override_reason,
                    "current": decision is current_decision,
                    "superseded": decision is not current_decision,
                }
                for decision in decisions
            ],
            "local_use": {
                "saved_session_count": len(sessions),
                "export_count": len(exports),
                "deployment_ready": False,
            },
            "retention": {
                **inventory,
                "active_plan": active_cleanup_plan,
                "receipts": [
                    _cleanup_receipt_view(receipt) for receipt in cleanup_receipts
                ],
            },
            "reproduction": {
                "active_plan": active_replay_plan,
                "derivations": [
                    _experiment_derivation_view(derivation)
                    for derivation in derivations
                    if derivation.reproduction_mode
                    is ReproductionMode.ADAPTED_REPRODUCTION
                ],
                "executions": _replay_execution_views(streams),
            },
            "stages": _stage_view(
                bool(projects),
                bool(datasets),
                bool(resolutions),
                len(artifacts) >= 2,
                bool(recommendations),
                current_decision is not None
                and current_decision.status
                in (UserDecisionStatus.SELECTED, UserDecisionStatus.PINNED),
            ),
        }

    def _artifact_view(
        self,
        store: TypedEvidenceStore,
        artifact: Artifact,
        candidate_key: str | None,
    ) -> dict[str, object]:
        view: dict[str, object] = {
            "key": candidate_key,
            "label": _candidate_label(candidate_key or "unknown"),
            "reference": record_reference(artifact).to_dict(),
        }
        try:
            if len(artifact.compatibility_groups) != 1:
                raise ApplicationServiceError("local_use_compatibility_ambiguous")
            model = store.read_record(artifact.base_model_revision).record
            group = store.read_record(artifact.compatibility_groups[0]).record
            run = store.read_record(artifact.producing_run).record
            if not isinstance(run, Run):
                raise ApplicationServiceError("local_use_dependency_invalid")
            target = store.read_record(run.execution_target).record
            if not isinstance(model, BaseModelRevision):
                raise ApplicationServiceError("local_use_dependency_invalid")
            if not isinstance(group, CompatibilityGroup):
                raise ApplicationServiceError("local_use_dependency_invalid")
            if not isinstance(target, ExecutionTarget):
                raise ApplicationServiceError("local_use_dependency_invalid")
            inspection = LocalUseService(self.project_root).inspect_artifact(
                artifact,
                model,
                group,
                target,
            )
        except EvidenceError as exc:
            view.update(
                {
                    "integrity_status": "failed",
                    "available": False,
                    "failure_code": exc.code,
                }
            )
        except ApplicationServiceError as exc:
            view.update(
                {
                    "integrity_status": "failed",
                    "available": False,
                    "failure_code": exc.code,
                }
            )
        else:
            view.update(inspection.to_view())
        return view


def _fixture_rendering_contract() -> ContentIdentity:
    return renderer_identity(
        FieldMapping("instruction", "response", "context"), RendererSpec()
    )


def _fixture_project_records() -> tuple[
    TaskDefinition,
    BaseModelRevision,
    Project,
    BaselinePolicy,
    ProjectPolicy,
]:
    task = TaskDefinition(
        task_id="task-fixture-runtime",
        display_name="Synthetic fixture rewrite",
        description="Rewrite synthetic local text for the offline fixture runtime.",
        input_schema={"required": ["instruction"]},
        output_schema={"required": ["response"]},
        rendering_contract=_fixture_rendering_contract(),
        objectives=("deterministic_rewrite",),
        capabilities=("text_generation",),
    )
    model = BaseModelRevision(
        model_id="model-fixture-runtime",
        display_name="Synthetic fixture model",
        model_family="fixture-family",
        architecture="fixture-causal-lm",
        source="public-fixture",
        revision="revision-one",
        weights_identity=_fixture_identity("weights"),
        tokenizer_identity=FixtureTokenizer.identity,
        license="Apache-2.0",
    )
    project = Project(
        project_id=FIXTURE_PROJECT_ID,
        display_name="Fixture runtime project",
        purpose="Exercise the deterministic offline Temper runtime.",
        task_definition=record_reference(task),
        base_model_revisions=(record_reference(model),),
    )
    baseline = BaselinePolicy(
        "baseline-fixture-runtime",
        (PerModelBaseline(_fixture_identity("comparison-policy")),),
    )
    policy = ProjectPolicy(
        policy_id="policy-fixture-runtime",
        project=record_reference(project),
        task_definition=record_reference(task),
        rendering_contract=task.rendering_contract,
        evaluation_policy=_fixture_identity("evaluation-policy"),
        case_suites=(_fixture_identity("case-suite"),),
        readiness_policy=_fixture_identity("readiness-policy"),
        retention_policy=_fixture_identity("retention-policy"),
        approved_recipe_families=("fixture",),
        baseline_policy=record_reference(baseline),
        recommendation_policy=_fixture_identity("recommendation-policy"),
    )
    return task, model, project, baseline, policy


def _dataset_request() -> DatasetImportRequest:
    return DatasetImportRequest(
        version_id=FIXTURE_DATASET_ID,
        field_mapping=FieldMapping("instruction", "response", "context"),
        renderer=RendererSpec(),
        filter_rule=FilterRule(1, 1000, 1000),
        deduplication_rule=DeduplicationRule(),
        split_rule=SplitRule(
            17,
            (SplitPart("train", 4), SplitPart("validation", 1)),
        ),
        tokenizer=FixtureTokenizer(),
        preview_limit=2,
    )


def _fixture_source_rows() -> list[dict[str, str]]:
    return [
        {
            "instruction": "Rewrite the synthetic alpha note",
            "context": "Alpha fixture context",
            "response": "Synthetic alpha rewrite",
        },
        {
            "instruction": "Rewrite the synthetic beta note",
            "context": "Beta fixture context",
            "response": "Synthetic beta rewrite",
        },
        {
            "instruction": "Rewrite the synthetic gamma note",
            "context": "Gamma fixture context",
            "response": "Synthetic gamma rewrite",
        },
    ]


def _fixture_hardware() -> tuple[HardwareRequirements, ExecutionTarget]:
    requirements = HardwareRequirements(
        requirements_id="requirements-fixture-runtime",
        execution_target_classes=("fixture_cpu",),
        accelerator_backends=("none",),
        minimum_accelerator_memory_bytes=0,
        minimum_system_memory_bytes=1,
        required_precision_modes=("fp32",),
        required_quantization_modes=(),
        required_capabilities=("fixture_adapter",),
        constraints={"local_only": True, "network_required": False},
    )
    target = ExecutionTarget(
        target_id="target-fixture-runtime",
        target_class="fixture_cpu",
        platform="portable",
        accelerator_backend="none",
        runtime_contract=_fixture_identity("fixture-runtime-contract"),
        capabilities=("fixture_adapter",),
        constraints={"local_only": True, "network_required": False},
    )
    return requirements, target


def _fixture_recipe_entries() -> tuple[RecipeCatalogEntry, RecipeCatalogEntry]:
    defaults = {
        "adapter_type": "lora",
        "target_modules": ["k_proj", "q_proj"],
        "rank": 4,
        "alpha": 8,
        "dropout": 0,
        "learning_rate": Decimal("0.0002"),
        "effective_batch_size": 2,
        "sequence_length": 128,
        "optimizer": "fixture_adamw",
        "precision": "fp32",
        "gradient_accumulation": 1,
        "seed": 17,
        "schedule": "linear",
        "training_steps": 4,
        "checkpoint_cadence": 2,
        "quantization": "none",
        "library_versions": {"fixture_runtime": "v1"},
    }
    balanced = Recipe(
        "recipe-fixture-runtime",
        "fixture",
        "v1",
        "deterministic",
        "small",
        "offline",
        "none",
        "fixture",
        "periodic",
        "selected_separately",
        "full",
        {},
    )
    capacity = Recipe(
        "recipe-fixture-challenger",
        "fixture",
        "v2",
        "deterministic",
        "medium",
        "offline",
        "none",
        "fixture",
        "periodic",
        "selected_separately",
        "full",
        {},
    )
    capacity_defaults = dict(defaults)
    capacity_defaults.update(
        {
            "rank": 8,
            "alpha": 16,
            "seed": 29,
            "training_steps": 6,
            "checkpoint_cadence": 3,
        }
    )
    return RecipeCatalogEntry(balanced, defaults), RecipeCatalogEntry(
        capacity, capacity_defaults
    )


def _fixture_suite(suite_id: str, kind: CaseSuiteKind) -> EvaluationSuite:
    return EvaluationSuite(
        suite_id=suite_id,
        kind=kind,
        state=SuiteEvidenceState.UNSEALED,
        cases=(
            EvaluationCase(
                f"case-{kind.value}-one",
                _fixture_identity(f"{kind.value}-case-one"),
            ),
            EvaluationCase(
                f"case-{kind.value}-two",
                _fixture_identity(f"{kind.value}-case-two"),
            ),
        ),
        evaluators=(
            EvaluatorSpec(
                "fixture-task-fit",
                EvaluatorKind.TASK_METRIC,
                "fixture_task_fit",
                MetricDirection.MAXIMIZE,
            ),
            EvaluatorSpec(
                "fixture-format-validity",
                EvaluatorKind.FORMAT_CHECK,
                "format_validity",
                MetricDirection.MAXIMIZE,
            ),
        ),
    )


def _fixture_recommendation_policy() -> RecommendationPolicy:
    return RecommendationPolicy(
        policy_id=FIXTURE_POLICY_ID,
        hard_qualifiers=(
            HardQualifier(
                "format_validity",
                ComparisonOperator.GREATER_THAN_OR_EQUAL,
                1,
            ),
        ),
        advisory_metrics=(),
        objectives=(
            OptimizationObjective(
                "fixture_task_fit", MetricDirection.MAXIMIZE, tie_tolerance=0
            ),
        ),
        baseline_comparisons=(),
        confidence_rules=(
            ConfidenceRule(
                ConfidenceLabel.LOW,
                (EvidenceStatus.PASSED,),
                (SuiteEvidenceState.SEALED,),
                2,
            ),
        ),
    )


def _current_suites(
    suites: tuple[EvaluationSuite, ...],
) -> tuple[EvaluationSuite, ...]:
    superseded = {
        suite.prior_suite.identity for suite in suites if suite.prior_suite is not None
    }
    return tuple(suite for suite in suites if suite.identity not in superseded)


def _current_reviews(reviews: tuple[Review, ...]) -> tuple[Review, ...]:
    superseded = {
        review.prior_review.identity
        for review in reviews
        if review.prior_review is not None
    }
    return tuple(review for review in reviews if review.identity not in superseded)


def _verified_checkpoint_identities(
    inventory: Mapping[str, object],
) -> dict[str, frozenset[str]]:
    """Index only checkpoint bytes whose observed identity matches run evidence."""

    values: dict[str, set[str]] = {}
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("byte_class") != "checkpoint":
            continue
        identity = entry.get("content_identity")
        subjects = entry.get("subjects")
        if (
            not isinstance(identity, Mapping)
            or not isinstance((identity_value := identity.get("value")), str)
            or not isinstance(subjects, list)
        ):
            continue
        for subject in subjects:
            if (
                isinstance(subject, Mapping)
                and subject.get("record_type") == "run"
                and isinstance((run_id := subject.get("logical_id")), str)
            ):
                values.setdefault(run_id, set()).add(identity_value)
    return {run_id: frozenset(items) for run_id, items in values.items()}


def _run_view(
    run: Run,
    events: tuple[Any, ...],
    verified_checkpoint_identities: frozenset[str],
) -> dict[str, object]:
    checkpoint_base_availability: dict[str, bool] = {}
    checkpoint_pending: dict[str, set[tuple[str, str]]] = {}
    checkpoint_removed: set[str] = set()
    for event in events:
        if event.event_type == "run_checkpoint":
            identity = event.payload.get("checkpoint_identity")
            if isinstance(identity, Mapping) and isinstance(identity.get("value"), str):
                checkpoint_base_availability[identity["value"]] = (
                    event.payload.get("resume_compatible") is True
                    and identity["value"] in verified_checkpoint_identities
                )
        elif event.event_type in {
            "run_checkpoint_cleanup_pending",
            "run_checkpoint_cleanup_cancelled",
            "run_checkpoint_removed",
        }:
            identity = event.payload.get("content_identity")
            if isinstance(identity, Mapping) and isinstance(identity.get("value"), str):
                identity_value = identity["value"]
                execution_id = event.payload.get("execution_id")
                entry_id = event.payload.get("entry_id")
                if not isinstance(execution_id, str) or not isinstance(entry_id, str):
                    checkpoint_removed.add(identity_value)
                    continue
                key = (execution_id, entry_id)
                pending = checkpoint_pending.setdefault(identity_value, set())
                if event.event_type == "run_checkpoint_cleanup_pending":
                    pending.add(key)
                elif key not in pending:
                    checkpoint_removed.add(identity_value)
                else:
                    pending.discard(key)
                    if event.event_type == "run_checkpoint_removed":
                        checkpoint_removed.add(identity_value)
    checkpoint_availability = {
        identity: available
        and identity not in checkpoint_removed
        and not checkpoint_pending.get(identity)
        for identity, available in checkpoint_base_availability.items()
    }
    terminal = next(
        (
            event.event_type.removeprefix("run_")
            for event in reversed(events)
            if event.event_type
            in {
                "run_preflight_blocked",
                "run_cancelled",
                "run_interrupted",
                "run_completed",
                "run_failed",
            }
        ),
        "running"
        if any(event.event_type == "run_launched" for event in events)
        else "unknown",
    )
    timeline: list[dict[str, object]] = []
    for event in events:
        item: dict[str, object] = {
            "sequence": event.sequence,
            "type": event.event_type,
        }
        if event.event_type == "run_progress":
            for field in ("step", "total_steps", "loss_microunits"):
                value = event.payload.get(field)
                if isinstance(value, int) and not isinstance(value, bool):
                    item[field] = value
        elif event.event_type == "run_log":
            code = event.payload.get("code")
            step = event.payload.get("step")
            if isinstance(code, str):
                item["code"] = code
            if isinstance(step, int) and not isinstance(step, bool):
                item["step"] = step
        elif event.event_type == "run_checkpoint":
            step = event.payload.get("step")
            byte_count = event.payload.get("byte_count")
            checkpoint_identity = event.payload.get("checkpoint_identity")
            if isinstance(step, int) and not isinstance(step, bool):
                item["step"] = step
            if isinstance(byte_count, int) and not isinstance(byte_count, bool):
                item["byte_count"] = byte_count
            if isinstance(checkpoint_identity, Mapping) and isinstance(
                checkpoint_identity.get("value"), str
            ):
                item["resume_available"] = checkpoint_availability.get(
                    checkpoint_identity["value"], False
                )
        elif event.event_type in {
            "run_checkpoint_cleanup_pending",
            "run_checkpoint_cleanup_cancelled",
            "run_checkpoint_removed",
        }:
            item["resume_available"] = event.payload.get("resume_available") is True
        timeline.append(item)
    checkpoint_events = tuple(
        event for event in events if event.event_type == "run_checkpoint"
    )
    available_checkpoints = sum(
        1
        for event in checkpoint_events
        if isinstance(event.payload.get("checkpoint_identity"), Mapping)
        and isinstance(event.payload["checkpoint_identity"].get("value"), str)
        and checkpoint_availability.get(
            event.payload["checkpoint_identity"]["value"], False
        )
    )
    return {
        "reference": record_reference(run).to_dict(),
        "run_id": run.run_id,
        "status": terminal,
        "attempt_number": run.attempt_number,
        "checkpoint_count": len(checkpoint_events),
        "resume_available_checkpoint_count": available_checkpoints,
        "events": timeline,
    }


def _cleanup_receipt_view(receipt: CleanupReceipt) -> dict[str, object]:
    return {
        "schema_version": "v1",
        "reference": record_reference(receipt).to_dict(),
        **receipt.to_payload(),
    }


def _experiment_derivation_view(
    derivation: ExperimentDerivation,
) -> dict[str, object]:
    return {
        "reference": record_reference(derivation).to_dict(),
        "mode": derivation.reproduction_mode.value,
        "reason_code": derivation.reason_code,
        "reason": derivation.reason,
        "source_experiment": record_reference(derivation.parent_experiment).to_dict(),
        "derived_experiment": record_reference(derivation.derived_experiment).to_dict(),
        "source_manifest_identity": {
            "algorithm": derivation.parent_experiment.manifest_identity.algorithm,
            "value": derivation.parent_experiment.manifest_identity.value,
        },
        "derived_manifest_identity": {
            "algorithm": derivation.derived_experiment.manifest_identity.algorithm,
            "value": derivation.derived_experiment.manifest_identity.value,
        },
        "manifest_changes": [
            change.to_dict() for change in derivation.manifest_diff.changes
        ],
    }


def _replay_execution_views(streams: tuple[Any, ...]) -> list[dict[str, object]]:
    views: list[dict[str, object]] = []
    for snapshot in streams:
        started = next(
            (
                event
                for event in snapshot.events
                if event.event_type == "replay_execution_started"
            ),
            None,
        )
        if started is None:
            continue
        terminal = next(
            (
                event
                for event in reversed(snapshot.events)
                if event.event_type
                in {
                    "replay_execution_completed",
                    "replay_execution_cancelled",
                    "replay_execution_interrupted",
                    "replay_execution_failed",
                }
            ),
            None,
        )
        mode = started.payload.get("mode")
        run_id = started.payload.get("run_id")
        views.append(
            {
                "mode": mode if isinstance(mode, str) else "unknown",
                "run_id": run_id if isinstance(run_id, str) else None,
                "status": (
                    str(terminal.payload.get("run_status", "failed"))
                    if terminal is not None
                    else "running"
                ),
                "exact_reproduction": mode == ReplayMode.STRICT.value,
                "adapted_reproduction": mode == ReplayMode.ADAPTED.value,
            }
        )
    return views


def _stage_view(
    project: bool,
    dataset: bool,
    resolutions: bool,
    runs: bool,
    evaluation: bool,
    selection: bool,
) -> list[dict[str, object]]:
    values = (
        ("setup", project),
        ("data", dataset),
        ("recipe", resolutions),
        ("run", runs),
        ("evaluate", evaluation),
        ("use", selection),
    )
    return [
        {
            "key": key,
            "complete": complete,
            "state": "complete" if complete else "pending",
        }
        for key, complete in values
    ]


def _empty_workspace() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "status": "empty",
        "fixture_mode": True,
        "offline": True,
        "general_chat": False,
        "external_dashboard": False,
        "hosted_deployment": False,
        "store": {
            "status": "verified",
            "record_count": 0,
            "record_counts": {},
            "event_stream_count": 0,
            "event_count": 0,
            "bundle_manifest_count": 0,
            "derived_state_rebuildable": True,
        },
        "project": None,
        "dataset": None,
        "resolutions": [],
        "runs": [],
        "artifacts": [],
        "evaluation": {"suites": [], "reviews": [], "results": []},
        "recommendation": None,
        "registry": [],
        "local_use": {
            "saved_session_count": 0,
            "export_count": 0,
            "deployment_ready": False,
        },
        "retention": {
            "schema_version": "v1",
            "retention_default": "full",
            "inventory_identity": None,
            "entry_count": 0,
            "logical_bytes": 0,
            "physical_bytes": 0,
            "reclaimable_physical_bytes": 0,
            "byte_classes": {},
            "entries": [],
            "active_plan": None,
            "receipts": [],
        },
        "reproduction": {
            "active_plan": None,
            "derivations": [],
            "executions": [],
        },
        "stages": _stage_view(False, False, False, False, False, False),
    }


def _fixture_identity(label: str) -> ContentIdentity:
    return ContentIdentity(
        "sha256", hashlib.sha256(f"temper-public-fixture:{label}".encode()).hexdigest()
    )


def _candidate_label(key: str) -> str:
    return {
        "ember": "Ember / balanced",
        "slate": "Slate / capacity",
    }.get(key, "Unknown fixture candidate")


def _logical_id(record: TypedRecord) -> str:
    for field in (
        "review_id",
        "session_id",
        "experiment_id",
        "export_id",
        "recommendation_id",
        "decision_id",
    ):
        value = getattr(record, field, None)
        if isinstance(value, str):
            return value
    raise ApplicationServiceError("fixture_logical_id_unsupported")
