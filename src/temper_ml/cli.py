"""Public-safe command-line access to Temper's local application services."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from typing import NoReturn

from temper_ml import __version__
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.fixture_journey import (
    FixtureJourneyService,
    FixtureTokenizer,
)
from temper_ml.app_services.experiments import (
    ReplayMode,
    adapted_replay_plan,
    strict_replay_plan,
)
from temper_ml.app_services.projects import ProjectService
from temper_ml.app_services.retention import RetentionService
from temper_ml.domain.experiments import (
    Experiment,
    ExperimentDerivation,
    ManifestDiff,
)
from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.projections import ContentIdentity, ProjectionError
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.runs import EvaluationMode
from temper_ml.domain.retention import CleanupReceipt
from temper_ml.runtime.paths import PortablePathError
from temper_ml.runtime.preflight import (
    EstimateComponents,
    PreflightError,
    estimate_resources,
    preflight,
)
from temper_ml.runtime.recipe_resolution import RecipeResolutionError, resolution_view
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.redaction import PublicSafetyError
from temper_ml.ui.server import serve_ui


_FixtureTokenizer = FixtureTokenizer


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
    cleanup_receipt = commands.add_parser("cleanup-receipt")
    cleanup_receipt.add_argument("project")
    cleanup_receipt.add_argument("--id", dest="receipt_id", required=True)
    cleanup_receipt.add_argument("--identity", dest="receipt_identity")
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
    inventory = commands.add_parser("inventory")
    inventory.add_argument("project")
    cleanup_plan = commands.add_parser("cleanup-plan")
    cleanup_plan.add_argument("project")
    cleanup_plan.add_argument("--entry-id", action="append", required=True)
    cleanup_execute = commands.add_parser("cleanup-execute")
    cleanup_execute.add_argument("project")
    cleanup_execute.add_argument("--entry-id", action="append", required=True)
    cleanup_execute.add_argument("--plan-id", required=True)
    cleanup_execute.add_argument("--execution-id", required=True)
    cleanup_execute.add_argument("--confirm", action="store_true")
    replay_plan = commands.add_parser("replay-plan")
    replay_plan.add_argument("project")
    replay_plan.add_argument("--experiment-id", required=True)
    replay_plan.add_argument("--experiment-identity")
    replay_plan.add_argument("--profile-id", required=True)
    replay_plan.add_argument("--profile-identity")
    replay_plan.add_argument(
        "--mode",
        choices=(ReplayMode.STRICT.value, ReplayMode.ADAPTED.value),
        default=ReplayMode.STRICT.value,
    )
    replay_plan.add_argument("--derivation-id")
    replay_plan.add_argument("--derivation-identity")
    for option in (
        "base-model-bytes",
        "adapter-optimizer-bytes",
        "peak-activation-bytes",
        "accelerator-runtime-overhead-bytes",
        "dataset-bytes",
        "host-runtime-overhead-bytes",
    ):
        replay_plan.add_argument(f"--{option}", type=int, required=True)
    fixture = commands.add_parser("fixture-workflow")
    fixture.add_argument("project")
    fixture.add_argument(
        "--evaluation-mode",
        choices=(EvaluationMode.NO_QUALITY_EVALUATION.value,),
        default=EvaluationMode.NO_QUALITY_EVALUATION.value,
    )
    ui = commands.add_parser("ui")
    ui.add_argument("project")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", default=8765, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "ui":
            serve_ui(arguments.project, host=arguments.host, port=arguments.port)
            return 0
        result = _run(arguments)
        encoded = dumps_canonical_json(result)
    except KeyboardInterrupt:
        return 130
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
    if arguments.command == "cleanup-receipt":
        store.verify()
        record = store.inspect_manifest(
            "cleanup_receipt",
            arguments.receipt_id,
            _optional_identity(arguments.receipt_identity),
        ).to_record()
        if not isinstance(record, CleanupReceipt):
            raise EvidenceError("cleanup_receipt_invalid")
        return {
            "schema_version": "v1",
            "command": "cleanup-receipt",
            "status": "available",
            "receipt": record.to_payload(),
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
        components = _estimate_components(arguments)
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
    if arguments.command == "inventory":
        result = RetentionService(arguments.project).inventory().to_view()
        result["command"] = "inventory"
        return result
    if arguments.command == "cleanup-plan":
        result = (
            RetentionService(arguments.project)
            .plan(tuple(arguments.entry_id))
            .to_view()
        )
        result["command"] = "cleanup-plan"
        return result
    if arguments.command == "cleanup-execute":
        service = RetentionService(arguments.project)
        plan = service.plan(
            tuple(arguments.entry_id), execution_id=arguments.execution_id
        )
        if plan.plan_id != arguments.plan_id:
            raise ApplicationServiceError("cleanup_plan_mismatch")
        receipt = service.execute(plan, confirm=arguments.confirm)
        return {
            "schema_version": "v1",
            "command": "cleanup-execute",
            "status": receipt.outcome.value,
            "receipt": receipt.to_payload(),
        }
    if arguments.command == "replay-plan":
        store.verify()
        source = store.inspect_manifest(
            "experiment",
            arguments.experiment_id,
            _optional_identity(arguments.experiment_identity),
        ).to_record()
        profile = store.inspect_manifest(
            "hardware_capability_profile",
            arguments.profile_id,
            _optional_identity(arguments.profile_identity),
        ).to_record()
        if not isinstance(source, Experiment) or not isinstance(
            profile, HardwareCapabilityProfile
        ):
            raise EvidenceError("replay_record_invalid")
        if arguments.mode == ReplayMode.STRICT.value:
            planned = source
            derivation = None
        else:
            if arguments.derivation_id is None:
                raise ApplicationServiceError("adapted_derivation_required")
            derivation_record = store.inspect_manifest(
                "experiment_derivation",
                arguments.derivation_id,
                _optional_identity(arguments.derivation_identity),
            ).to_record()
            if (
                not isinstance(derivation_record, ExperimentDerivation)
                or derivation_record.parent_experiment != source
            ):
                raise EvidenceError("replay_record_invalid")
            derivation = derivation_record
            planned = derivation.derived_experiment
        resolution_record = store.read_record(planned.recipe_resolution).record
        requirements_record = store.read_record(planned.hardware_requirements).record
        target_record = store.read_record(planned.execution_target).record
        if not isinstance(resolution_record, RecipeResolution):
            raise EvidenceError("replay_record_invalid")
        if not isinstance(requirements_record, HardwareRequirements):
            raise EvidenceError("replay_record_invalid")
        if not isinstance(target_record, ExecutionTarget):
            raise EvidenceError("replay_record_invalid")
        replay_preflight = preflight(
            resolution_record,
            requirements_record,
            target_record,
            profile,
            estimate_resources(resolution_record, _estimate_components(arguments)),
        )
        replay_plan_result = (
            strict_replay_plan(source, replay_preflight)
            if derivation is None
            else adapted_replay_plan(derivation, replay_preflight)
        )
        value = replay_plan_result.to_view()
        value["schema_version"] = "v1"
        value["command"] = "replay-plan"
        return value
    if arguments.command == "fixture-workflow":
        return _fixture_workflow(
            arguments.project,
            evaluation_mode=EvaluationMode(arguments.evaluation_mode),
        )
    raise EvidenceError("unknown_command")


def _fixture_workflow(
    project_root: str,
    *,
    evaluation_mode: EvaluationMode,
) -> dict[str, object]:
    return FixtureJourneyService(project_root).legacy_workflow(evaluation_mode)


def _estimate_components(arguments: argparse.Namespace) -> EstimateComponents:
    return EstimateComponents(
        base_model_bytes=arguments.base_model_bytes,
        adapter_optimizer_bytes=arguments.adapter_optimizer_bytes,
        peak_activation_bytes=arguments.peak_activation_bytes,
        accelerator_runtime_overhead_bytes=(
            arguments.accelerator_runtime_overhead_bytes
        ),
        dataset_bytes=arguments.dataset_bytes,
        host_runtime_overhead_bytes=arguments.host_runtime_overhead_bytes,
    )


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
