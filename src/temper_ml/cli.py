"""Public-safe command-line access to a local Temper evidence store."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from decimal import Decimal
import hashlib
import sys
from typing import NoReturn

from temper_ml import __version__
from temper_ml.app_services.datasets import (
    DatasetImportRequest,
    DatasetService,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.experiments import (
    ExperimentFreezeRequest,
    ExperimentService,
)
from temper_ml.app_services.local_use import (
    AdapterExportRequest,
    LocalUseRequest,
    LocalUseService,
)
from temper_ml.app_services.projects import ProjectService
from temper_ml.app_services.projects import ProjectCreateRequest
from temper_ml.app_services.runs import RunLaunchRequest, RunService
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import (
    CompatibilityGroup,
    RuntimeTargetConstraint,
)
from temper_ml.domain.datasets import (
    DeduplicationRule,
    FieldMapping,
    FilterRule,
    RendererSpec,
    SplitPart,
    SplitRule,
    renderer_identity,
)
from temper_ml.domain.experiments import ManifestDiff
from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.projections import ContentIdentity, ProjectionError
from temper_ml.domain.policies import BaselinePolicy, PerModelBaseline
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import record_reference
from temper_ml.domain.runs import EvaluationMode
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.runtime.paths import PortablePathError
from temper_ml.runtime.preflight import (
    EstimateComponents,
    PreflightError,
    estimate_resources,
    preflight,
    capture_capability_profile,
)
from temper_ml.runtime.recipe_resolution import (
    RecipeCatalog,
    RecipeCatalogEntry,
    RecipeResolutionError,
    RecipeResolver,
    resolution_view,
)
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.redaction import PublicSafetyError


class _JsonArgumentParser(argparse.ArgumentParser):
    """Argparse variant whose failures cannot disclose user-supplied values."""

    def error(self, message: str) -> NoReturn:
        del message
        _emit_json(sys.stderr, {"status": "error", "code": "usage_error"})
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(prog="temper")
    parser.add_argument(
        "--version", action="version", version=f"temper-ml {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "verify", "dump"):
        command = commands.add_parser(name)
        command.add_argument("project")
    manifest = commands.add_parser("manifest")
    manifest.add_argument("project")
    manifest.add_argument("--type", dest="record_type", required=True)
    manifest.add_argument("--id", dest="logical_id", required=True)
    manifest.add_argument("--identity")
    project_status = commands.add_parser("project-status")
    project_status.add_argument("project")
    project_status.add_argument("--id", dest="project_id", required=True)
    project_status.add_argument("--identity", dest="project_identity")
    project_status.add_argument("--policy-id")
    project_status.add_argument("--policy-identity")
    recipe_resolution = commands.add_parser("recipe-resolution")
    recipe_resolution.add_argument("project")
    recipe_resolution.add_argument("--id", dest="resolution_id", required=True)
    recipe_resolution.add_argument("--identity", dest="resolution_identity")
    manifest_diff = commands.add_parser("manifest-diff")
    manifest_diff.add_argument("project")
    manifest_diff.add_argument("--id", dest="diff_id", required=True)
    manifest_diff.add_argument("--identity", dest="diff_identity")
    preflight_command = commands.add_parser("preflight")
    preflight_command.add_argument("project")
    preflight_command.add_argument("--resolution-id", required=True)
    preflight_command.add_argument("--resolution-identity")
    preflight_command.add_argument("--profile-id", required=True)
    preflight_command.add_argument("--profile-identity")
    for option in (
        "base-model-bytes",
        "adapter-optimizer-bytes",
        "peak-activation-bytes",
        "accelerator-runtime-overhead-bytes",
        "dataset-bytes",
        "host-runtime-overhead-bytes",
    ):
        preflight_command.add_argument(f"--{option}", type=int, required=True)
    fixture = commands.add_parser("fixture-workflow")
    fixture.add_argument("project")
    fixture.add_argument(
        "--evaluation-mode",
        choices=(EvaluationMode.NO_QUALITY_EVALUATION.value,),
        default=EvaluationMode.NO_QUALITY_EVALUATION.value,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = _run(arguments)
        encoded = dumps_canonical_json(result)
    except PublicSafetyError as exc:
        _emit_error(exc.code)
        return 3
    except (
        ApplicationServiceError,
        PortablePathError,
        PreflightError,
        RecipeResolutionError,
    ) as exc:
        _emit_error(exc.code)
        return 1
    except EvidenceError as exc:
        _emit_error(exc.code)
        return 3 if exc.code.startswith("admission_") else 1
    except (OSError, UnicodeError):
        _emit_error("filesystem_error")
        return 4
    except Exception:
        _emit_error("internal_error")
        return 4
    try:
        _write_bytes(sys.stdout, encoded)
    except (OSError, UnicodeError):
        _emit_error("filesystem_error")
        return 4
    return 0


def _run(arguments: argparse.Namespace) -> object:
    store = TypedEvidenceStore(arguments.project)
    if arguments.command in {"status", "verify"}:
        result = store.verify().to_dict()
        result["schema_version"] = "v1"
        result["command"] = arguments.command
        return result
    if arguments.command == "dump":
        return store.public_dump().value
    if arguments.command == "manifest":
        identity = (
            _parse_identity(arguments.identity)
            if arguments.identity is not None
            else None
        )
        return store.inspect_manifest(
            arguments.record_type,
            arguments.logical_id,
            identity,
        ).to_dict()
    if arguments.command == "project-status":
        opened = ProjectService(arguments.project).open(
            arguments.project_id,
            project_identity=_optional_identity(arguments.project_identity),
            policy_id=arguments.policy_id,
            policy_identity=_optional_identity(arguments.policy_identity),
        )
        value = opened.to_view()
        value["schema_version"] = "v1"
        value["command"] = "project-status"
        return value
    if arguments.command == "recipe-resolution":
        store.verify()
        record = store.inspect_manifest(
            "recipe_resolution",
            arguments.resolution_id,
            _optional_identity(arguments.resolution_identity),
        ).to_record()
        if not isinstance(record, RecipeResolution):
            raise EvidenceError("recipe_resolution_invalid")
        value = resolution_view(record)
        value["schema_version"] = "v1"
        value["command"] = "recipe-resolution"
        return value
    if arguments.command == "manifest-diff":
        store.verify()
        record = store.inspect_manifest(
            "manifest_diff",
            arguments.diff_id,
            _optional_identity(arguments.diff_identity),
        ).to_record()
        if not isinstance(record, ManifestDiff):
            raise EvidenceError("manifest_diff_invalid")
        return {
            "schema_version": "v1",
            "command": "manifest-diff",
            "status": "available",
            "identity": {
                "algorithm": record.identity.algorithm,
                "value": record.identity.value,
            },
            "diff": record.to_payload(),
        }
    if arguments.command == "preflight":
        store.verify()
        resolution_record = store.inspect_manifest(
            "recipe_resolution",
            arguments.resolution_id,
            _optional_identity(arguments.resolution_identity),
        ).to_record()
        profile_record = store.inspect_manifest(
            "hardware_capability_profile",
            arguments.profile_id,
            _optional_identity(arguments.profile_identity),
        ).to_record()
        if not isinstance(resolution_record, RecipeResolution) or not isinstance(
            profile_record, HardwareCapabilityProfile
        ):
            raise EvidenceError("preflight_record_invalid")
        requirements_record = store.read_record(
            resolution_record.hardware_requirements
        ).record
        target_record = store.read_record(resolution_record.execution_target).record
        if not isinstance(requirements_record, HardwareRequirements):
            raise EvidenceError("preflight_record_invalid")
        if not isinstance(target_record, ExecutionTarget):
            raise EvidenceError("preflight_record_invalid")
        components = EstimateComponents(
            base_model_bytes=arguments.base_model_bytes,
            adapter_optimizer_bytes=arguments.adapter_optimizer_bytes,
            peak_activation_bytes=arguments.peak_activation_bytes,
            accelerator_runtime_overhead_bytes=(
                arguments.accelerator_runtime_overhead_bytes
            ),
            dataset_bytes=arguments.dataset_bytes,
            host_runtime_overhead_bytes=arguments.host_runtime_overhead_bytes,
        )
        result = preflight(
            resolution_record,
            requirements_record,
            target_record,
            profile_record,
            estimate_resources(resolution_record, components),
        ).to_view()
        result["schema_version"] = "v1"
        result["command"] = "preflight"
        return result
    if arguments.command == "fixture-workflow":
        return _fixture_workflow(
            arguments.project,
            evaluation_mode=EvaluationMode(arguments.evaluation_mode),
        )
    raise EvidenceError("unknown_command")


class _FixtureTokenizer:
    identity = ContentIdentity(
        "sha256", hashlib.sha256(b"temper-public-fixture-tokenizer-v1").hexdigest()
    )

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.encode("utf-8"))


def _fixture_workflow(
    project_root: str,
    *,
    evaluation_mode: EvaluationMode,
) -> dict[str, object]:
    """Exercise the complete synthetic Slice 5 workflow without external I/O."""

    mapping = FieldMapping("instruction", "response", "context")
    renderer = RendererSpec()
    rendering_contract = renderer_identity(mapping, renderer)
    task = TaskDefinition(
        task_id="task-fixture-runtime",
        display_name="Synthetic fixture rewrite",
        description="Rewrite synthetic local text for the offline fixture runtime.",
        input_schema={"required": ["instruction"]},
        output_schema={"required": ["response"]},
        rendering_contract=rendering_contract,
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
        tokenizer_identity=_FixtureTokenizer.identity,
        license="Apache-2.0",
    )
    project = Project(
        project_id="project-fixture-runtime",
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
    opened = ProjectService(project_root).create(
        ProjectCreateRequest(task, project, baseline, policy, (model,))
    )
    dataset_request = DatasetImportRequest(
        version_id="dataset-fixture-runtime",
        field_mapping=mapping,
        renderer=renderer,
        filter_rule=FilterRule(1, 1000, 1000),
        deduplication_rule=DeduplicationRule(),
        split_rule=SplitRule(
            17,
            (SplitPart("train", 4), SplitPart("validation", 1)),
        ),
        tokenizer=_FixtureTokenizer(),
        preview_limit=2,
    )
    source_rows = [
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
    prepared = DatasetService(project_root).import_json(
        dumps_canonical_json(source_rows), dataset_request
    )
    recipe = Recipe(
        recipe_id="recipe-fixture-runtime",
        family="fixture",
        version="v1",
        training_profile="deterministic",
        adapter_size="small",
        memory_mode="offline",
        quantization="none",
        training_duration="fixture",
        checkpoint_policy="periodic",
        evaluation_intensity="selected_separately",
        retention_policy="standard",
        expert_overrides={},
    )
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
    catalog = RecipeCatalog(
        (
            RecipeCatalogEntry(
                recipe,
                {
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
                },
            ),
        )
    )
    entry = catalog.select("fixture", "v1")
    resolution = RecipeResolver().resolve(
        entry,
        base_model_revision=model,
        hardware_requirements=requirements,
        execution_target=target,
    )
    group = CompatibilityGroup(
        group_id="group-fixture-runtime",
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
                ("fixture_adapter",),
            ),
        ),
        merge_methods=(),
    )
    experiment = ExperimentService(project_root).freeze(
        ExperimentFreezeRequest(
            experiment_id="experiment-fixture-runtime",
            opened_project=opened,
            dataset_version=prepared.version.identity,
            base_model_revision=model,
            recipe=entry.recipe,
            recipe_resolution=resolution,
            compatibility_group=group,
            hardware_requirements=requirements,
            execution_target=target,
        )
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
    estimate = estimate_resources(
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
    launch_request = RunLaunchRequest(
        run_id="run-fixture-runtime",
        request_id="request-fixture-runtime",
        artifact_id="artifact-fixture-runtime",
        experiment=experiment,
        recipe_resolution=resolution,
        prepared_dataset=prepared,
        base_model_revision=model,
        compatibility_group=group,
        hardware_requirements=requirements,
        execution_target=target,
        hardware_capability_profile=profile,
        estimate=estimate,
        evaluation_mode=evaluation_mode,
    )
    run_service = RunService(project_root)
    try:
        run = run_service.reopen_completed(launch_request)
    except ApplicationServiceError as exc:
        if exc.code != "run_not_found":
            raise
        run = run_service.launch(launch_request)
    if run.artifact is None:
        raise ApplicationServiceError("fixture_workflow_artifact_missing")
    local = LocalUseService(project_root)
    focused = local.focused(
        LocalUseRequest(
            artifact=run.artifact,
            base_model_revision=model,
            compatibility_group=group,
            execution_target=target,
            settings=_fixture_inference_settings(),
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
            settings=_fixture_inference_settings(),
            inputs=(
                {"text": "Synthetic batch prompt one"},
                {"text": "Synthetic batch prompt two"},
            ),
        )
    )
    exported = local.export(
        AdapterExportRequest(
            export_id="export-fixture-runtime",
            artifact=run.artifact,
            base_model_revision=model,
            compatibility_group=group,
            execution_target=target,
        )
    )
    local.verify_export(
        exported.record,
        artifact=run.artifact,
        base_model_revision=model,
        compatibility_group=group,
        execution_target=target,
    )
    store = TypedEvidenceStore(project_root)
    verification = store.verify()
    public = store.public_dump().value
    return {
        "schema_version": "v1",
        "command": "fixture-workflow",
        "status": "verified",
        "evaluation_mode": evaluation_mode.value,
        "project": record_reference(project).to_dict(),
        "dataset_version": record_reference(prepared.version).to_dict(),
        "recipe_resolution": record_reference(resolution).to_dict(),
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
        "public_projection_verified": public["classification"] == "public_projection",
        "hosted_deployment": False,
        "deployment_ready": False,
    }


def _fixture_identity(label: str) -> ContentIdentity:
    return ContentIdentity(
        "sha256", hashlib.sha256(f"temper-public-fixture:{label}".encode()).hexdigest()
    )


def _fixture_inference_settings():
    from temper_ml.runtime.fixture_inference import InferenceSettings

    return InferenceSettings(temperature=0, maximum_tokens=32, seed=17)


def _parse_identity(value: str) -> ContentIdentity:
    digest = value.removeprefix("sha256:")
    try:
        return ContentIdentity("sha256", digest)
    except ProjectionError:
        raise EvidenceError("invalid_identity") from None


def _optional_identity(value: str | None) -> ContentIdentity | None:
    return _parse_identity(value) if value is not None else None


def _emit_error(code: str) -> None:
    _emit_json(sys.stderr, {"status": "error", "code": code})


def _emit_json(stream: object, value: object) -> None:
    _write_bytes(stream, dumps_canonical_json(value))


def _write_bytes(stream: object, value: bytes) -> None:
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write(value)
        return
    write = getattr(stream, "write")
    write(value.decode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
