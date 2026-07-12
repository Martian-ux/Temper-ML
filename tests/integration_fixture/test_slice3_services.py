from decimal import Decimal
import hashlib
from pathlib import Path

from temper_ml.app_services.experiments import (
    ExperimentFreezeRequest,
    ExperimentService,
    ReplayMode,
    plan_replay,
)
from temper_ml.app_services.projects import ProjectCreateRequest, ProjectService
from temper_ml.cli import main
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import CompatibilityGroup, RuntimeTargetConstraint
from temper_ml.domain.experiments import ReproductionMode
from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.policies import BaselinePolicy, PerModelBaseline
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe
from temper_ml.domain.records import record_reference
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.runtime.preflight import (
    EstimateComponents,
    capture_capability_profile,
    estimate_resources,
    preflight,
)
from temper_ml.runtime.recipe_resolution import (
    RecipeCatalog,
    RecipeCatalogEntry,
    RecipeResolver,
    ResolutionConstraint,
)
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json


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


def _assert_canonical(output: str) -> dict[str, object]:
    value = loads_canonical_json(output.encode())
    assert isinstance(value, dict)
    assert output == dumps_canonical_json(value).decode()
    return value


def test_slice_three_services_form_one_deterministic_canonical_workflow(
    tmp_path: Path, capsys
) -> None:
    task = TaskDefinition(
        task_id="task-slice-three",
        display_name="Synthetic rewrite",
        description="Rewrite synthetic text while preserving named entities.",
        input_schema={"required": ["input"]},
        output_schema={"required": ["output"]},
        rendering_contract=_identity("renderer"),
        objectives=("preserve_entities",),
        capabilities=("text_generation",),
    )
    model = BaseModelRevision(
        model_id="model-slice-three",
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
        project_id="project-slice-three",
        display_name="Slice three project",
        purpose="Exercise deterministic project and experiment services.",
        task_definition=record_reference(task),
        base_model_revisions=(record_reference(model),),
    )
    baseline = BaselinePolicy(
        "baseline-slice-three", (PerModelBaseline(_identity("comparison")),)
    )
    policy = ProjectPolicy(
        policy_id="policy-slice-three",
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
        recipe_id="recipe-slice-three",
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
        requirements_id="requirements-slice-three",
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
        target_id="target-slice-three",
        target_class="wsl2_rocm",
        platform="linux",
        accelerator_backend="rocm",
        runtime_contract=_identity("runtime-contract"),
        capabilities=("lora",),
        constraints={"local_only": True},
    )
    catalog = RecipeCatalog(
        (
            RecipeCatalogEntry(
                recipe,
                _defaults(),
                ("rank", "sequence_length"),
            ),
        )
    )
    entry = catalog.with_expert_overrides(
        "balanced", "v1", {"rank": 16, "sequence_length": 768}
    )
    resolver = RecipeResolver()
    resolution = resolver.resolve(
        entry,
        base_model_revision=model,
        hardware_requirements=requirements,
        execution_target=target,
    )
    repeated_resolution = resolver.resolve(
        entry,
        base_model_revision=model,
        hardware_requirements=requirements,
        execution_target=target,
    )
    assert resolution.identity == repeated_resolution.identity
    assert resolution.to_payload() == repeated_resolution.to_payload()

    group = CompatibilityGroup(
        group_id="group-slice-three",
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
    experiment_service = ExperimentService(tmp_path)
    experiment = experiment_service.freeze(
        ExperimentFreezeRequest(
            experiment_id="experiment-slice-three",
            opened_project=opened,
            dataset_version=_identity("dataset-version"),
            base_model_revision=model,
            recipe=entry.recipe,
            recipe_resolution=resolution,
            compatibility_group=group,
            hardware_requirements=requirements,
            execution_target=target,
        )
    )
    profile = capture_capability_profile(
        profile_id="profile-slice-three",
        execution_target=target,
        accelerator_backend="rocm",
        accelerator_architecture="synthetic-arch",
        accelerator_model="Synthetic accelerator",
        accelerator_count=1,
        accelerator_memory_bytes=(8_000,),
        system_memory_bytes=16_000,
        supported_precision_modes=("bf16",),
        supported_quantization_modes=(),
        capabilities=("lora",),
        library_versions={"fixture_runtime": "v1"},
    )
    experiment_service.store.write_record(profile)
    components = EstimateComponents(2_000, 500, 1_000, 500, 1_000, 2_000)
    initial_preflight = preflight(
        resolution,
        requirements,
        target,
        profile,
        estimate_resources(resolution, components),
    )
    assert initial_preflight.ready

    constrained_requirements = HardwareRequirements(
        requirements_id="requirements-slice-three-adapted",
        execution_target_classes=("wsl2_rocm",),
        accelerator_backends=("rocm",),
        minimum_accelerator_memory_bytes=2_000,
        minimum_system_memory_bytes=8_000,
        required_precision_modes=("bf16",),
        required_quantization_modes=(),
        required_capabilities=("lora",),
        constraints={"local_only": True},
    )
    constrained_resolution = resolver.resolve(
        RecipeCatalogEntry(entry.recipe, _defaults(), ("rank", "sequence_length")),
        base_model_revision=model,
        hardware_requirements=constrained_requirements,
        execution_target=target,
        constraints=(
            ResolutionConstraint("memory_budget", {"rank": 4, "sequence_length": 256}),
        ),
    )
    derivation = experiment_service.clone(
        experiment,
        experiment_id="experiment-slice-three-adapted",
        replacements={
            "recipe_resolution": record_reference(constrained_resolution),
            "hardware_requirements": record_reference(constrained_requirements),
        },
        derivation_id="derivation-slice-three-adapted",
        diff_id="diff-slice-three-adapted",
        reason_code="hardware_adaptation",
        reason="The original resolved manifest exceeded the available memory.",
        reproduction_mode=ReproductionMode.ADAPTED_REPRODUCTION,
        supporting_records=(constrained_requirements, constrained_resolution),
    )
    low_profile = capture_capability_profile(
        profile_id="profile-slice-three-low-memory",
        execution_target=target,
        accelerator_backend="rocm",
        accelerator_architecture="synthetic-arch",
        accelerator_model="Synthetic accelerator",
        accelerator_count=1,
        accelerator_memory_bytes=(3_000,),
        system_memory_bytes=16_000,
        supported_precision_modes=("bf16",),
        supported_quantization_modes=(),
        capabilities=("lora",),
        library_versions={"fixture_runtime": "v1"},
    )
    strict_blocked = preflight(
        resolution,
        requirements,
        target,
        low_profile,
        estimate_resources(resolution, components),
    )
    adapted_components = EstimateComponents(1_500, 200, 400, 300, 1_000, 2_000)
    adapted_ready = preflight(
        constrained_resolution,
        constrained_requirements,
        target,
        low_profile,
        estimate_resources(constrained_resolution, adapted_components),
    )
    replay = plan_replay(
        experiment,
        strict_blocked,
        adapted_derivation=derivation,
        adapted_preflight=adapted_ready,
    )
    assert replay.ready
    assert replay.mode is ReplayMode.ADAPTED
    assert replay.source_experiment.identity == experiment.identity
    assert replay.planned_experiment.identity != experiment.identity

    verification = experiment_service.store.verify()
    assert verification.record_counts["experiment"] == 2
    assert verification.record_counts["manifest_diff"] == 1
    assert verification.record_counts["experiment_derivation"] == 1

    assert (
        main(
            [
                "project-status",
                str(tmp_path),
                "--id",
                project.project_id,
                "--policy-id",
                policy.policy_id,
            ]
        )
        == 0
    )
    project_view = _assert_canonical(capsys.readouterr().out)
    assert project_view["status"] == "open"

    assert (
        main(
            [
                "recipe-resolution",
                str(tmp_path),
                "--id",
                resolution.resolution_id,
                "--identity",
                f"sha256:{resolution.identity.value}",
            ]
        )
        == 0
    )
    resolution_view = _assert_canonical(capsys.readouterr().out)
    assert resolution_view["status"] == "resolved"
    assert resolution_view["manifest"] == resolution.to_payload()

    assert (
        main(
            [
                "preflight",
                str(tmp_path),
                "--resolution-id",
                resolution.resolution_id,
                "--profile-id",
                profile.profile_id,
                "--base-model-bytes",
                "2000",
                "--adapter-optimizer-bytes",
                "500",
                "--peak-activation-bytes",
                "1000",
                "--accelerator-runtime-overhead-bytes",
                "500",
                "--dataset-bytes",
                "1000",
                "--host-runtime-overhead-bytes",
                "2000",
            ]
        )
        == 0
    )
    preflight_view = _assert_canonical(capsys.readouterr().out)
    assert preflight_view["status"] == "ready"

    assert (
        main(
            [
                "manifest-diff",
                str(tmp_path),
                "--id",
                derivation.manifest_diff.diff_id,
                "--identity",
                f"sha256:{derivation.manifest_diff.identity.value}",
            ]
        )
        == 0
    )
    diff_view = _assert_canonical(capsys.readouterr().out)
    assert diff_view["status"] == "available"
    assert [change["path"] for change in diff_view["diff"]["changes"]] == [
        "/hardware_requirements/identity/value",
        "/hardware_requirements/logical_id",
        "/recipe_resolution/identity/value",
        "/recipe_resolution/logical_id",
    ]
    assert (
        str(tmp_path)
        not in dumps_canonical_json(
            [project_view, resolution_view, preflight_view, diff_view]
        ).decode()
    )
