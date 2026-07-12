"""Experiment freeze, clone, and explicit replay-planning services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

from temper_ml.app_services._records import (
    require_no_conflicting_logical_revision,
    write_record_idempotently,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.projects import OpenedProject
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import (
    CompatibilityGroup,
    check_runtime_target_compatibility,
)
from temper_ml.domain.experiments import (
    Experiment,
    ExperimentDerivation,
    ReproductionMode,
    derive_experiment,
)
from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import (
    RecordValidationError,
    TypedRecord,
    record_reference,
)
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.runtime.preflight import PreflightResult
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest


@dataclass(frozen=True)
class ExperimentFreezeRequest:
    """Exact records and identities used to freeze one scientific intention."""

    experiment_id: str
    opened_project: OpenedProject
    dataset_version: ContentIdentity
    base_model_revision: BaseModelRevision
    recipe: Recipe
    recipe_resolution: RecipeResolution
    compatibility_group: CompatibilityGroup
    hardware_requirements: HardwareRequirements
    execution_target: ExecutionTarget


class ReplayMode(str, Enum):
    STRICT = "strict_replay"
    ADAPTED = "adapted_reproduction"


class ReplayStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ReplayPlan:
    """A non-canonical plan that never mutates its source experiment."""

    plan_id: str
    mode: ReplayMode
    status: ReplayStatus
    source_experiment: Experiment
    planned_experiment: Experiment
    preflight: PreflightResult
    reasons: tuple[str, ...]
    derivation: ExperimentDerivation | None = None

    @property
    def ready(self) -> bool:
        return self.status is ReplayStatus.READY

    def to_view(self) -> dict[str, object]:
        value: dict[str, object] = {
            "plan_id": self.plan_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "source_experiment": record_reference(self.source_experiment).to_dict(),
            "planned_experiment": record_reference(self.planned_experiment).to_dict(),
            "source_manifest_identity": {
                "algorithm": self.source_experiment.manifest_identity.algorithm,
                "value": self.source_experiment.manifest_identity.value,
            },
            "planned_manifest_identity": {
                "algorithm": self.planned_experiment.manifest_identity.algorithm,
                "value": self.planned_experiment.manifest_identity.value,
            },
            "reasons": list(self.reasons),
            "preflight": self.preflight.to_view(),
        }
        if self.derivation is not None:
            value["derivation"] = record_reference(self.derivation).to_dict()
            value["manifest_diff"] = record_reference(
                self.derivation.manifest_diff
            ).to_dict()
        return value


class ExperimentService:
    """Persist immutable experiments and checked derivation evidence."""

    def __init__(self, project_root: Path | str) -> None:
        self.store = TypedEvidenceStore(project_root)

    def freeze(self, request: ExperimentFreezeRequest) -> Experiment:
        """Freeze and persist one fully resolved experiment idempotently."""

        if not isinstance(request, ExperimentFreezeRequest):
            raise ApplicationServiceError("invalid_experiment_request")
        if not isinstance(request.opened_project, OpenedProject):
            raise ApplicationServiceError("invalid_experiment_request")
        if not isinstance(request.dataset_version, ContentIdentity):
            raise ApplicationServiceError("dataset_version_invalid")
        if (
            not isinstance(request.opened_project.project, Project)
            or not isinstance(request.opened_project.task_definition, TaskDefinition)
            or not isinstance(request.opened_project.project_policy, ProjectPolicy)
        ):
            raise ApplicationServiceError("opened_project_store_mismatch")
        self._require_exact_stored_records(
            (
                request.opened_project.project,
                request.opened_project.task_definition,
                request.opened_project.project_policy,
            ),
            mismatch_code="opened_project_store_mismatch",
        )
        experiment = Experiment(
            experiment_id=request.experiment_id,
            project=record_reference(request.opened_project.project),
            project_policy=record_reference(request.opened_project.project_policy),
            task_definition=record_reference(request.opened_project.task_definition),
            dataset_version=request.dataset_version,
            base_model_revision=record_reference(request.base_model_revision),
            tokenizer_identity=request.base_model_revision.tokenizer_identity,
            recipe=record_reference(request.recipe),
            recipe_resolution=record_reference(request.recipe_resolution),
            evaluation_policy=request.opened_project.project_policy.evaluation_policy,
            compatibility_group=record_reference(request.compatibility_group),
            hardware_requirements=record_reference(request.hardware_requirements),
            execution_target=record_reference(request.execution_target),
        )
        self._validate_experiment_dependencies(
            experiment,
            project=request.opened_project.project,
            policy=request.opened_project.project_policy,
            task=request.opened_project.task_definition,
            base_model=request.base_model_revision,
            recipe=request.recipe,
            resolution=request.recipe_resolution,
            group=request.compatibility_group,
            requirements=request.hardware_requirements,
            target=request.execution_target,
        )
        require_no_conflicting_logical_revision(
            self.store,
            experiment,
            conflict_code="experiment_revision_conflict",
        )
        for record in (
            request.base_model_revision,
            request.recipe,
            request.hardware_requirements,
            request.execution_target,
            request.recipe_resolution,
            request.compatibility_group,
            experiment,
        ):
            write_record_idempotently(
                self.store,
                record,
                conflict_code="experiment_dependency_conflict",
            )
        self.store.append_event(
            "experiment-lifecycle",
            EventRequest(
                f"experiment-frozen-{experiment.experiment_id}",
                "experiment_frozen",
                {
                    "experiment_manifest_frozen": True,
                    "execution_target_explicit": True,
                },
            ),
        )
        self.store.verify()
        return experiment

    def clone(
        self,
        parent: Experiment,
        *,
        experiment_id: str,
        replacements: Mapping[str, object],
        derivation_id: str,
        diff_id: str,
        reason_code: str,
        reason: str,
        reproduction_mode: ReproductionMode = ReproductionMode.SCIENTIFIC_DERIVATION,
        supporting_records: tuple[TypedRecord, ...] = (),
    ) -> ExperimentDerivation:
        """Clone a parent and persist its exact material manifest difference."""

        if not isinstance(parent, Experiment):
            raise ApplicationServiceError("parent_experiment_invalid")
        if not isinstance(replacements, Mapping):
            raise ApplicationServiceError("experiment_replacements_invalid")
        allowed = {field.name for field in fields(Experiment)} - {"experiment_id"}
        if not replacements or set(replacements) - allowed:
            raise ApplicationServiceError("experiment_replacements_invalid")
        if not isinstance(supporting_records, tuple) or any(
            not isinstance(record, TypedRecord) for record in supporting_records
        ):
            raise ApplicationServiceError("supporting_records_invalid")
        self._require_exact_stored_records(
            (parent,),
            mismatch_code="parent_experiment_store_mismatch",
        )
        replacement_values: dict[str, Any] = dict(replacements)
        try:
            derived = replace(
                parent,
                experiment_id=experiment_id,
                **replacement_values,
            )
            evidence = derive_experiment(
                parent,
                derived,
                derivation_id=derivation_id,
                diff_id=diff_id,
                reason_code=reason_code,
                reason=reason,
                reproduction_mode=reproduction_mode,
            )
        except (RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError("experiment_derivation_invalid") from None

        for output_record in (derived, evidence.manifest_diff, evidence):
            require_no_conflicting_logical_revision(
                self.store,
                output_record,
                conflict_code="experiment_derivation_conflict",
            )
        for supporting_record in supporting_records:
            write_record_idempotently(
                self.store,
                supporting_record,
                conflict_code="experiment_dependency_conflict",
            )
        self._validate_stored_experiment(derived)
        for record in (derived, evidence.manifest_diff, evidence):
            write_record_idempotently(
                self.store,
                record,
                conflict_code="experiment_derivation_conflict",
            )
        self.store.append_event(
            "experiment-lifecycle",
            EventRequest(
                f"experiment-derived-{derived.experiment_id}",
                "experiment_derived",
                {
                    "derived_experiment_created": True,
                    "adapted_reproduction": (
                        reproduction_mode is ReproductionMode.ADAPTED_REPRODUCTION
                    ),
                    "manifest_diff_recorded": True,
                },
            ),
        )
        self.store.verify()
        return evidence

    def _require_exact_stored_records(
        self,
        records: tuple[TypedRecord, ...],
        *,
        mismatch_code: str,
    ) -> None:
        try:
            stored_records = tuple(
                self.store.read_record(record_reference(record)) for record in records
            )
        except (EvidenceError, RecordValidationError, TypeError, ValueError):
            raise ApplicationServiceError(mismatch_code) from None
        if any(
            type(stored.record) is not type(expected)
            or stored.envelope.to_dict() != expected.to_dict()
            for stored, expected in zip(stored_records, records, strict=True)
        ):
            raise ApplicationServiceError(mismatch_code)

    def _validate_stored_experiment(self, experiment: Experiment) -> None:
        records = {
            "project": self.store.read_record(experiment.project).record,
            "policy": self.store.read_record(experiment.project_policy).record,
            "task": self.store.read_record(experiment.task_definition).record,
            "base_model": self.store.read_record(experiment.base_model_revision).record,
            "recipe": self.store.read_record(experiment.recipe).record,
            "resolution": self.store.read_record(experiment.recipe_resolution).record,
            "group": self.store.read_record(experiment.compatibility_group).record,
            "requirements": self.store.read_record(
                experiment.hardware_requirements
            ).record,
            "target": self.store.read_record(experiment.execution_target).record,
        }
        expected_types = {
            "project": Project,
            "policy": ProjectPolicy,
            "task": TaskDefinition,
            "base_model": BaseModelRevision,
            "recipe": Recipe,
            "resolution": RecipeResolution,
            "group": CompatibilityGroup,
            "requirements": HardwareRequirements,
            "target": ExecutionTarget,
        }
        if any(
            not isinstance(records[name], expected)
            for name, expected in expected_types.items()
        ):
            raise ApplicationServiceError("experiment_dependency_invalid")
        self._validate_experiment_dependencies(
            experiment,
            project=records["project"],
            policy=records["policy"],
            task=records["task"],
            base_model=records["base_model"],
            recipe=records["recipe"],
            resolution=records["resolution"],
            group=records["group"],
            requirements=records["requirements"],
            target=records["target"],
        )

    @staticmethod
    def _validate_experiment_dependencies(
        experiment: Experiment,
        *,
        project: object,
        policy: object,
        task: object,
        base_model: object,
        recipe: object,
        resolution: object,
        group: object,
        requirements: object,
        target: object,
    ) -> None:
        if not isinstance(project, Project) or not isinstance(policy, ProjectPolicy):
            raise ApplicationServiceError("experiment_project_invalid")
        if not isinstance(task, TaskDefinition) or not isinstance(
            base_model, BaseModelRevision
        ):
            raise ApplicationServiceError("experiment_model_invalid")
        if not isinstance(recipe, Recipe) or not isinstance(
            resolution, RecipeResolution
        ):
            raise ApplicationServiceError("experiment_recipe_invalid")
        if not isinstance(group, CompatibilityGroup):
            raise ApplicationServiceError("experiment_compatibility_invalid")
        if not isinstance(requirements, HardwareRequirements) or not isinstance(
            target, ExecutionTarget
        ):
            raise ApplicationServiceError("experiment_hardware_invalid")
        if experiment.project != record_reference(project):
            raise ApplicationServiceError("experiment_project_mismatch")
        if experiment.project_policy != record_reference(policy):
            raise ApplicationServiceError("experiment_project_mismatch")
        if experiment.task_definition != record_reference(task):
            raise ApplicationServiceError("experiment_task_mismatch")
        if policy.project != record_reference(
            project
        ) or policy.task_definition != record_reference(task):
            raise ApplicationServiceError("experiment_project_mismatch")
        if record_reference(base_model) not in project.base_model_revisions:
            raise ApplicationServiceError("experiment_model_not_in_project")
        if experiment.base_model_revision != record_reference(base_model):
            raise ApplicationServiceError("experiment_model_mismatch")
        if experiment.tokenizer_identity != base_model.tokenizer_identity:
            raise ApplicationServiceError("experiment_tokenizer_mismatch")
        if recipe.family not in policy.approved_recipe_families:
            raise ApplicationServiceError("recipe_family_not_approved")
        if experiment.recipe != record_reference(recipe):
            raise ApplicationServiceError("experiment_recipe_mismatch")
        if experiment.recipe_resolution != record_reference(resolution):
            raise ApplicationServiceError("experiment_resolution_mismatch")
        if resolution.recipe != record_reference(recipe):
            raise ApplicationServiceError("experiment_resolution_mismatch")
        if resolution.base_model_revision != record_reference(base_model):
            raise ApplicationServiceError("experiment_resolution_mismatch")
        if resolution.hardware_requirements != record_reference(requirements):
            raise ApplicationServiceError("experiment_resolution_mismatch")
        if resolution.execution_target != record_reference(target):
            raise ApplicationServiceError("experiment_resolution_mismatch")
        if experiment.evaluation_policy != policy.evaluation_policy:
            raise ApplicationServiceError("experiment_evaluation_policy_mismatch")
        if experiment.compatibility_group != record_reference(group):
            raise ApplicationServiceError("experiment_compatibility_mismatch")
        if group.base_model_revision != record_reference(base_model):
            raise ApplicationServiceError("experiment_compatibility_mismatch")
        if group.tokenizer_identity != base_model.tokenizer_identity:
            raise ApplicationServiceError("experiment_compatibility_mismatch")
        if group.rendering_template != task.rendering_contract:
            raise ApplicationServiceError("experiment_rendering_mismatch")
        if (
            group.adapter_type != resolution.adapter_type
            or group.target_modules != resolution.target_modules
        ):
            raise ApplicationServiceError("experiment_compatibility_mismatch")
        if experiment.hardware_requirements != record_reference(requirements):
            raise ApplicationServiceError("experiment_hardware_mismatch")
        if experiment.execution_target != record_reference(target):
            raise ApplicationServiceError("experiment_hardware_mismatch")
        if (
            target.target_class not in requirements.execution_target_classes
            or target.accelerator_backend not in requirements.accelerator_backends
        ):
            raise ApplicationServiceError("experiment_hardware_mismatch")
        if not check_runtime_target_compatibility(group, target).compatible:
            raise ApplicationServiceError("runtime_target_incompatible")


def strict_replay_plan(
    source: Experiment,
    preflight: PreflightResult,
) -> ReplayPlan:
    """Plan the original manifest unchanged, even when that plan is blocked."""

    if not isinstance(source, Experiment):
        raise ApplicationServiceError("source_experiment_invalid")
    if not isinstance(preflight, PreflightResult):
        raise ApplicationServiceError("preflight_invalid")
    reasons: list[str] = []
    if source.recipe_resolution != record_reference(preflight.resolution):
        reasons.append("recipe_resolution_changed")
    if source.hardware_requirements != record_reference(preflight.requirements):
        reasons.append("hardware_requirements_changed")
    if source.execution_target != record_reference(preflight.target):
        reasons.append("execution_target_changed")
    reasons.extend(preflight.blocking_reasons)
    return _replay_plan(
        ReplayMode.STRICT,
        source,
        source,
        preflight,
        tuple(dict.fromkeys(reasons)),
        None,
    )


def adapted_replay_plan(
    derivation: ExperimentDerivation,
    preflight: PreflightResult,
) -> ReplayPlan:
    """Plan a visibly derived adapted reproduction and its exact manifest diff."""

    if not isinstance(derivation, ExperimentDerivation) or (
        derivation.reproduction_mode is not ReproductionMode.ADAPTED_REPRODUCTION
    ):
        raise ApplicationServiceError("adapted_derivation_invalid")
    if not isinstance(preflight, PreflightResult):
        raise ApplicationServiceError("preflight_invalid")
    reasons: list[str] = []
    if derivation.derived_experiment.recipe_resolution != record_reference(
        preflight.resolution
    ):
        reasons.append("adapted_resolution_mismatch")
    if derivation.derived_experiment.hardware_requirements != record_reference(
        preflight.requirements
    ):
        reasons.append("adapted_requirements_mismatch")
    if derivation.derived_experiment.execution_target != record_reference(
        preflight.target
    ):
        reasons.append("adapted_target_mismatch")
    reasons.extend(preflight.blocking_reasons)
    return _replay_plan(
        ReplayMode.ADAPTED,
        derivation.parent_experiment,
        derivation.derived_experiment,
        preflight,
        tuple(dict.fromkeys(reasons)),
        derivation,
    )


def plan_replay(
    source: Experiment,
    strict_preflight: PreflightResult,
    *,
    adapted_derivation: ExperimentDerivation | None = None,
    adapted_preflight: PreflightResult | None = None,
) -> ReplayPlan:
    """Prefer exact replay; require an explicit derived experiment if it is blocked."""

    strict = strict_replay_plan(source, strict_preflight)
    if strict.ready:
        return strict
    if adapted_derivation is None and adapted_preflight is None:
        return _replay_plan(
            ReplayMode.STRICT,
            source,
            source,
            strict_preflight,
            tuple((*strict.reasons, "adaptation_required")),
            None,
        )
    if adapted_derivation is None or adapted_preflight is None:
        raise ApplicationServiceError("adapted_replay_incomplete")
    if adapted_derivation.parent_experiment.identity != source.identity:
        raise ApplicationServiceError("adapted_parent_mismatch")
    return adapted_replay_plan(adapted_derivation, adapted_preflight)


def _replay_plan(
    mode: ReplayMode,
    source: Experiment,
    planned: Experiment,
    preflight: PreflightResult,
    reasons: tuple[str, ...],
    derivation: ExperimentDerivation | None,
) -> ReplayPlan:
    identity_input: dict[str, Any] = {
        "mode": mode.value,
        "source": record_reference(source).to_dict(),
        "planned": record_reference(planned).to_dict(),
        "profile": record_reference(preflight.profile).to_dict(),
        "resolution": record_reference(preflight.resolution).to_dict(),
        "target": record_reference(preflight.target).to_dict(),
        "preflight": preflight.to_view(),
        "reasons": list(reasons),
    }
    if derivation is not None:
        identity_input["derivation"] = record_reference(derivation).to_dict()
    digest = hashlib.sha256(dumps_canonical_json(identity_input)).hexdigest()
    return ReplayPlan(
        plan_id=f"replay-{digest}",
        mode=mode,
        status=ReplayStatus.READY if not reasons else ReplayStatus.BLOCKED,
        source_experiment=source,
        planned_experiment=planned,
        preflight=preflight,
        reasons=reasons,
        derivation=derivation,
    )
