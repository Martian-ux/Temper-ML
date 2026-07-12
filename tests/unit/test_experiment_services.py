from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
from pathlib import Path

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.experiments import (
    ExperimentFreezeRequest,
    ExperimentService,
    ReplayMode,
    plan_replay,
)
from temper_ml.app_services.projects import (
    OpenedProject,
    ProjectCreateRequest,
    ProjectService,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import CompatibilityGroup, RuntimeTargetConstraint
from temper_ml.domain.experiments import Experiment, ReproductionMode
from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.policies import BaselinePolicy, PerModelBaseline
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import record_reference
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.runtime.preflight import (
    EstimateComponents,
    PreflightResult,
    capture_capability_profile,
    estimate_resources,
    preflight,
)
from temper_ml.runtime.recipe_resolution import (
    RecipeCatalogEntry,
    RecipeResolver,
    ResolutionConstraint,
)


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _defaults() -> dict[str, object]:
    return {
        "adapter_type": "lora",
        "target_modules": ["k_proj", "q_proj"],
        "rank": 8,
        "alpha": 16,
        "dropout": 0,
        "learning_rate": Decimal("0.0002"),
        "effective_batch_size": 8,
        "sequence_length": 512,
        "optimizer": "adamw",
        "precision": "bf16",
        "gradient_accumulation": 2,
        "seed": 7,
        "schedule": "linear",
        "training_steps": 20,
        "checkpoint_cadence": 5,
        "quantization": "none",
        "library_versions": {"fixture_runtime": "v1"},
    }


@dataclass(frozen=True)
class _Fixture:
    opened: OpenedProject
    model: BaseModelRevision
    recipe: Recipe
    requirements: HardwareRequirements
    target: ExecutionTarget
    resolution: RecipeResolution
    group: CompatibilityGroup
    experiment: Experiment
    service: ExperimentService


def _fixture(tmp_path: Path) -> _Fixture:
    task = TaskDefinition(
        task_id="task-rewrite",
        display_name="Synthetic rewrite",
        description="Rewrite synthetic text without changing entities.",
        input_schema={"required": ["input"]},
        output_schema={"required": ["output"]},
        rendering_contract=_identity("renderer"),
        objectives=("preserve_entities",),
        capabilities=("text_generation",),
    )
    model = BaseModelRevision(
        model_id="model-synthetic",
        display_name="Synthetic model",
        model_family="fixture-family",
        architecture="fixture-causal-lm",
        source="public-fixture",
        revision="revision-one",
        weights_identity=_identity("weights"),
        tokenizer_identity=_identity("tokenizer"),
        license="Apache-2.0",
    )
    project = Project(
        project_id="project-rewrite",
        display_name="Rewrite project",
        purpose="Train one synthetic rewrite adapter.",
        task_definition=record_reference(task),
        base_model_revisions=(record_reference(model),),
    )
    baseline = BaselinePolicy(
        "baseline-default", (PerModelBaseline(_identity("comparison")),)
    )
    policy = ProjectPolicy(
        policy_id="policy-default",
        project=record_reference(project),
        task_definition=record_reference(task),
        rendering_contract=task.rendering_contract,
        evaluation_policy=_identity("evaluation"),
        case_suites=(_identity("regression"),),
        readiness_policy=_identity("readiness"),
        retention_policy=_identity("retention"),
        approved_recipe_families=("balanced",),
        baseline_policy=record_reference(baseline),
        recommendation_policy=_identity("recommendation"),
    )
    opened = ProjectService(tmp_path).create(
        ProjectCreateRequest(task, project, baseline, policy, (model,))
    )
    recipe = Recipe(
        recipe_id="recipe-balanced",
        family="balanced",
        version="v1",
        training_profile="balanced",
        adapter_size="small",
        memory_mode="standard",
        quantization="none",
        training_duration="fixture",
        checkpoint_policy="periodic",
        evaluation_intensity="normal",
        retention_policy="standard",
        expert_overrides={},
    )
    requirements = HardwareRequirements(
        requirements_id="requirements-balanced",
        execution_target_classes=("wsl2_rocm",),
        accelerator_backends=("rocm",),
        minimum_accelerator_memory_bytes=4_000,
        minimum_system_memory_bytes=8_000,
        required_precision_modes=("bf16",),
        required_quantization_modes=(),
        required_capabilities=("lora",),
        constraints={"local_only": True},
    )
    target = ExecutionTarget(
        target_id="target-wsl2-rocm",
        target_class="wsl2_rocm",
        platform="linux",
        accelerator_backend="rocm",
        runtime_contract=_identity("runtime-contract"),
        capabilities=("lora",),
        constraints={"local_only": True},
    )
    entry = RecipeCatalogEntry(recipe, _defaults(), ("rank", "sequence_length"))
    resolution = RecipeResolver().resolve(
        entry,
        base_model_revision=model,
        hardware_requirements=requirements,
        execution_target=target,
    )
    group = CompatibilityGroup(
        group_id="group-synthetic",
        base_model_revision=record_reference(model),
        tokenizer_identity=model.tokenizer_identity,
        rendering_template=task.rendering_contract,
        adapter_type=resolution.adapter_type,
        target_modules=resolution.target_modules,
        runtime_targets=(
            RuntimeTargetConstraint(
                target.target_class,
                target.accelerator_backend,
                target.runtime_contract,
                ("lora",),
            ),
        ),
        merge_methods=("linear",),
    )
    service = ExperimentService(tmp_path)
    experiment = service.freeze(
        ExperimentFreezeRequest(
            experiment_id="experiment-original",
            opened_project=opened,
            dataset_version=_identity("dataset-version"),
            base_model_revision=model,
            recipe=recipe,
            recipe_resolution=resolution,
            compatibility_group=group,
            hardware_requirements=requirements,
            execution_target=target,
        )
    )
    return _Fixture(
        opened,
        model,
        recipe,
        requirements,
        target,
        resolution,
        group,
        experiment,
        service,
    )


def _preflight(
    resolution: RecipeResolution,
    requirements: HardwareRequirements,
    target: ExecutionTarget,
    *,
    memory: int,
    adapted: bool = False,
    runtime_version: str = "v1",
) -> PreflightResult:
    profile = capture_capability_profile(
        profile_id=f"profile-{memory}-{'adapted' if adapted else 'strict'}",
        execution_target=target,
        accelerator_backend="rocm",
        accelerator_architecture="synthetic-arch",
        accelerator_model="Synthetic accelerator",
        accelerator_count=1,
        accelerator_memory_bytes=(memory,),
        system_memory_bytes=16_000,
        supported_precision_modes=("bf16",),
        supported_quantization_modes=(),
        capabilities=("lora",),
        library_versions={"fixture_runtime": runtime_version},
    )
    components = (
        EstimateComponents(1_500, 200, 400, 300, 1_000, 2_000)
        if adapted
        else EstimateComponents(2_000, 500, 1_000, 500, 1_000, 2_000)
    )
    return preflight(
        resolution,
        requirements,
        target,
        profile,
        estimate_resources(resolution, components),
    )


def test_freeze_is_idempotent_and_binds_every_resolved_dependency(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    repeated = fixture.service.freeze(
        ExperimentFreezeRequest(
            experiment_id=fixture.experiment.experiment_id,
            opened_project=fixture.opened,
            dataset_version=fixture.experiment.dataset_version,
            base_model_revision=fixture.model,
            recipe=fixture.recipe,
            recipe_resolution=fixture.resolution,
            compatibility_group=fixture.group,
            hardware_requirements=fixture.requirements,
            execution_target=fixture.target,
        )
    )

    assert repeated.identity == fixture.experiment.identity
    verification = fixture.service.store.verify()
    assert verification.record_counts["experiment"] == 1
    assert verification.event_count == 2


def test_freeze_rejects_an_opened_project_from_another_store_without_mutation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "source")
    destination_root = tmp_path / "destination"
    destination = ExperimentService(destination_root)
    assert not destination_root.exists()

    with pytest.raises(ApplicationServiceError) as error:
        destination.freeze(
            ExperimentFreezeRequest(
                experiment_id="experiment-cross-store",
                opened_project=fixture.opened,
                dataset_version=_identity("cross-store-dataset"),
                base_model_revision=fixture.model,
                recipe=fixture.recipe,
                recipe_resolution=fixture.resolution,
                compatibility_group=fixture.group,
                hardware_requirements=fixture.requirements,
                execution_target=fixture.target,
            )
        )

    assert error.value.code == "opened_project_store_mismatch"
    assert not destination_root.exists()
    assert destination.store.iter_records() == ()
    assert destination.store.iter_streams() == ()


def test_clone_rejects_a_parent_from_another_store_without_supporting_leaks(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "source")
    destination_root = tmp_path / "destination"
    destination = ExperimentService(destination_root)
    assert not destination_root.exists()

    with pytest.raises(ApplicationServiceError) as error:
        destination.clone(
            fixture.experiment,
            experiment_id="experiment-cross-store-clone",
            replacements={"dataset_version": _identity("changed-dataset")},
            derivation_id="derivation-cross-store",
            diff_id="diff-cross-store",
            reason_code="dataset_update",
            reason="Use a different synthetic dataset revision.",
            supporting_records=(fixture.requirements,),
        )

    assert error.value.code == "parent_experiment_store_mismatch"
    assert not destination_root.exists()
    assert destination.store.iter_records() == ()
    assert destination.store.iter_streams() == ()


def test_changed_machine_uses_strict_unchanged_or_explicit_adapted_derivation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    strict_ready = _preflight(
        fixture.resolution,
        fixture.requirements,
        fixture.target,
        memory=8_000,
    )
    strict_plan = plan_replay(fixture.experiment, strict_ready)
    assert strict_plan.ready
    assert strict_plan.mode is ReplayMode.STRICT
    assert strict_plan.planned_experiment.identity == fixture.experiment.identity
    assert (
        strict_plan.planned_experiment.manifest_identity
        == fixture.experiment.manifest_identity
    )

    strict_blocked = _preflight(
        fixture.resolution,
        fixture.requirements,
        fixture.target,
        memory=3_000,
    )
    blocked_plan = plan_replay(fixture.experiment, strict_blocked)
    assert not blocked_plan.ready
    assert blocked_plan.mode is ReplayMode.STRICT
    assert blocked_plan.reasons[-1] == "adaptation_required"

    version_drift = _preflight(
        fixture.resolution,
        fixture.requirements,
        fixture.target,
        memory=8_000,
        runtime_version="v2",
    )
    version_plan = plan_replay(fixture.experiment, version_drift)
    assert not version_plan.ready
    assert "resolved_library_versions_match" in version_plan.reasons
    assert version_plan.reasons[-1] == "adaptation_required"

    adapted_requirements = HardwareRequirements(
        requirements_id="requirements-adapted",
        execution_target_classes=("wsl2_rocm",),
        accelerator_backends=("rocm",),
        minimum_accelerator_memory_bytes=2_000,
        minimum_system_memory_bytes=8_000,
        required_precision_modes=("bf16",),
        required_quantization_modes=(),
        required_capabilities=("lora",),
        constraints={"local_only": True},
    )
    adapted_resolution = RecipeResolver().resolve(
        RecipeCatalogEntry(
            fixture.recipe,
            _defaults(),
            ("rank", "sequence_length"),
        ),
        base_model_revision=fixture.model,
        hardware_requirements=adapted_requirements,
        execution_target=fixture.target,
        constraints=(
            ResolutionConstraint("memory_budget", {"rank": 4, "sequence_length": 256}),
        ),
    )
    derivation = fixture.service.clone(
        fixture.experiment,
        experiment_id="experiment-adapted",
        replacements={
            "recipe_resolution": record_reference(adapted_resolution),
            "hardware_requirements": record_reference(adapted_requirements),
        },
        derivation_id="derivation-adapted",
        diff_id="diff-adapted",
        reason_code="hardware_adaptation",
        reason="The original resolved manifest exceeded the available memory.",
        reproduction_mode=ReproductionMode.ADAPTED_REPRODUCTION,
        supporting_records=(adapted_requirements, adapted_resolution),
    )
    adapted_ready = _preflight(
        adapted_resolution,
        adapted_requirements,
        fixture.target,
        memory=3_000,
        adapted=True,
    )
    adapted_plan = plan_replay(
        fixture.experiment,
        strict_blocked,
        adapted_derivation=derivation,
        adapted_preflight=adapted_ready,
    )

    assert adapted_plan.ready
    assert adapted_plan.mode is ReplayMode.ADAPTED
    assert adapted_plan.planned_experiment.identity != fixture.experiment.identity
    assert (
        derivation.manifest_diff.apply(fixture.experiment.scientific_manifest())
        == derivation.derived_experiment.scientific_manifest()
    )
    assert fixture.service.store.verify().record_counts["experiment_derivation"] == 1


def test_replay_plan_identity_includes_estimate_and_constraint_content(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first_preflight = _preflight(
        fixture.resolution,
        fixture.requirements,
        fixture.target,
        memory=8_000,
    )
    second_preflight = preflight(
        fixture.resolution,
        fixture.requirements,
        fixture.target,
        first_preflight.profile,
        replace(
            first_preflight.estimate,
            accelerator_memory_bytes=(
                first_preflight.estimate.accelerator_memory_bytes - 1
            ),
        ),
    )

    first_plan = plan_replay(fixture.experiment, first_preflight)
    second_plan = plan_replay(fixture.experiment, second_preflight)

    assert first_plan.ready and second_plan.ready
    assert first_plan.reasons == second_plan.reasons == ()
    assert first_preflight.profile.identity == second_preflight.profile.identity
    assert first_plan.plan_id != second_plan.plan_id
    assert first_plan.to_view()["preflight"] != second_plan.to_view()["preflight"]


def test_clone_rejects_a_logical_rename_without_material_manifest_change(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(
        ApplicationServiceError, match="experiment_replacements_invalid"
    ):
        fixture.service.clone(
            fixture.experiment,
            experiment_id="experiment-renamed",
            replacements={},
            derivation_id="derivation-renamed",
            diff_id="diff-renamed",
            reason_code="rename_only",
            reason="Only the logical name changed.",
        )


def test_freeze_allows_distinct_expert_override_revisions_of_one_recipe(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    overridden_recipe = replace(fixture.recipe, expert_overrides={"rank": 16})
    overridden_resolution = RecipeResolver().resolve(
        RecipeCatalogEntry(overridden_recipe, _defaults(), ("rank",)),
        base_model_revision=fixture.model,
        hardware_requirements=fixture.requirements,
        execution_target=fixture.target,
    )

    second = fixture.service.freeze(
        ExperimentFreezeRequest(
            experiment_id="experiment-expert-override",
            opened_project=fixture.opened,
            dataset_version=fixture.experiment.dataset_version,
            base_model_revision=fixture.model,
            recipe=overridden_recipe,
            recipe_resolution=overridden_resolution,
            compatibility_group=fixture.group,
            hardware_requirements=fixture.requirements,
            execution_target=fixture.target,
        )
    )

    assert second.recipe.logical_id == fixture.experiment.recipe.logical_id
    assert second.recipe.identity != fixture.experiment.recipe.identity
    assert fixture.service.store.verify().record_counts["recipe"] == 2
