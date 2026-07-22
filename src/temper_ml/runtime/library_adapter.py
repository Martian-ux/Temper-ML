"""Temper-normalized library training and inference adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from temper_ml.domain.artifacts import build_bytes_bundle_manifest
from temper_ml.domain.datasets import DatasetVersion
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.hardware import ExecutionTarget, HardwareCapabilityProfile
from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    freeze_json_object,
    identity_fields,
    record_reference,
    require_identifier,
    thaw_json,
)
from temper_ml.runtime.controller import (
    RuntimeControllerError,
    SerializedResourceCoordinator,
    SerializedRunController,
)
from temper_ml.runtime.fixture_adapter import (
    FixtureAdapter,
    FixtureAdapterOutput,
    FixtureAdapterRequest,
    FixtureCheckpoint,
    FixtureControl,
    FixtureLog,
    FixtureProgress,
    FixtureTermination,
)
from temper_ml.runtime.fixture_inference import (
    FixtureInferenceRequest,
    FixtureInferenceResult,
)
from temper_ml.runtime.library_backend import (
    LibraryBackend,
    LibraryCapability,
    LibraryCheckpointPayload,
    LibraryExecutionContext,
    LibraryInferenceResult,
    LibraryRuntimeError,
)
from temper_ml.runtime.paths import PortableLocation
from temper_ml.runtime.preflight import capture_capability_profile
from temper_ml.runtime.protocol import (
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperation,
)
from temper_ml.runtime.staging import (
    StagingError,
    TransferDirection,
    build_transfer_manifest,
    read_verified_transfer,
    stage_transfer,
)
from temper_ml.store.canonical_json import dumps_canonical_json

LIBRARY_RUNTIME_PROJECTION = HashProjection("runtime.library_adapter", "v1")
LIBRARY_TRAINING_STATE_PROJECTION = HashProjection(
    "runtime.library_training_state", "v1"
)
LIBRARY_INFERENCE_EVIDENCE_PROJECTION = HashProjection(
    "runtime.library_inference_evidence", "v1"
)
LIBRARY_INFERENCE_REQUEST_PROJECTION = HashProjection(
    "runtime.library_inference_request", "v1"
)


class LibraryAdapterExecutionError(LibraryRuntimeError):
    """A failed run plus its already-validated boundary evidence."""

    def __init__(
        self,
        code: str,
        messages: tuple[RuntimeMessage, ...],
        receipts: tuple[object, ...],
    ) -> None:
        from temper_ml.runtime.staging import TransferReceipt

        if not isinstance(messages, tuple) or any(
            not isinstance(message, RuntimeMessage) for message in messages
        ):
            raise LibraryRuntimeError("library_execution_error_invalid")
        if not isinstance(receipts, tuple) or any(
            not isinstance(receipt, TransferReceipt) for receipt in receipts
        ):
            raise LibraryRuntimeError("library_execution_error_invalid")
        self.messages = messages
        self.receipts = receipts
        super().__init__(code)


def library_training_state_identity(
    experiment: Experiment,
    resolution: RecipeResolution,
    dataset_version: DatasetVersion,
    runtime_identity: ContentIdentity,
    step: int,
) -> ContentIdentity:
    """Project one portable library training state without worker-owned IDs."""

    if (
        not isinstance(experiment, Experiment)
        or not isinstance(resolution, RecipeResolution)
        or not isinstance(dataset_version, DatasetVersion)
        or not isinstance(runtime_identity, ContentIdentity)
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or step > resolution.training_steps
    ):
        raise LibraryRuntimeError("library_training_state_invalid")
    return content_identity(
        LIBRARY_TRAINING_STATE_PROJECTION,
        {
            "schema_version": "v1",
            "experiment": record_reference(experiment).to_dict(),
            "recipe_resolution": record_reference(resolution).to_dict(),
            "dataset_version_identity": identity_fields(dataset_version.identity),
            "runtime_identity": identity_fields(runtime_identity),
            "step": step,
        },
    )


@dataclass(frozen=True)
class LibraryRuntimeSources:
    """Private local sources supplied to one worker runtime instance."""

    model_source: Path
    tokenizer_source: Path
    staging_root: Path
    target_class: str
    base_model_revision: RecordReference
    tokenizer_identity: ContentIdentity

    def __post_init__(self) -> None:
        for field in ("model_source", "tokenizer_source", "staging_root"):
            value = getattr(self, field)
            if not isinstance(value, Path) or not value.is_absolute():
                raise LibraryRuntimeError("library_runtime_source_invalid")
        try:
            require_identifier("target_class", self.target_class)
        except RecordValidationError:
            raise LibraryRuntimeError("library_target_class_invalid") from None
        if (
            not isinstance(self.base_model_revision, RecordReference)
            or self.base_model_revision.record_type != "base_model_revision"
            or not isinstance(self.tokenizer_identity, ContentIdentity)
        ):
            raise LibraryRuntimeError("library_runtime_source_identity_invalid")


def library_runtime_identity(
    capability: LibraryCapability,
    sources: LibraryRuntimeSources,
) -> ContentIdentity:
    """Bind the executable library/runtime surface without private paths."""

    if not isinstance(capability, LibraryCapability) or not isinstance(
        sources, LibraryRuntimeSources
    ):
        raise LibraryRuntimeError("library_runtime_identity_invalid")
    return content_identity(
        LIBRARY_RUNTIME_PROJECTION,
        {
            "schema_version": "v1",
            "runtime": "temper_library_adapter",
            "target_class": sources.target_class,
            "protocol": "v1",
            "library_versions": dict(sorted(capability.library_versions.items())),
            "capabilities": list(capability.capabilities),
        },
    )


class LibraryAdapter(FixtureAdapter):
    """Run real library machinery while preserving Temper artifact contracts."""

    runtime_kind = "library"

    def __init__(
        self,
        backend: LibraryBackend,
        sources: LibraryRuntimeSources,
        *,
        capability: LibraryCapability | None = None,
        resources: SerializedResourceCoordinator | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        interruption_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if not isinstance(backend, LibraryBackend):
            raise LibraryRuntimeError("library_backend_invalid")
        if not isinstance(sources, LibraryRuntimeSources):
            raise LibraryRuntimeError("library_runtime_source_invalid")
        self.backend = backend
        self.sources = sources
        observed_capability = backend.probe()
        if not isinstance(observed_capability, LibraryCapability):
            raise LibraryRuntimeError("library_capability_invalid")
        if capability is not None and capability != observed_capability:
            raise LibraryRuntimeError("library_capability_mismatch")
        self.capability = observed_capability
        resource_names = tuple(
            f"accelerator-{index}" for index in range(self.capability.accelerator_count)
        ) or ("cpu-runtime",)
        self.resources = (
            resources
            if resources is not None
            else SerializedResourceCoordinator({name: 1 for name in resource_names})
        )
        if not isinstance(self.resources, SerializedResourceCoordinator):
            raise LibraryRuntimeError("library_resource_coordinator_invalid")
        for callback in (
            cancellation_requested,
            interruption_requested,
            progress_callback,
        ):
            if callback is not None and not callable(callback):
                raise LibraryRuntimeError("library_control_callback_invalid")
        self._cancellation_requested = cancellation_requested or (lambda: False)
        self._interruption_requested = interruption_requested or (lambda: False)
        self._progress_callback = progress_callback
        self.resource_names = resource_names
        self._active_run_ids: set[str] = set()
        self._active_run_lock = RLock()
        self.runtime_identity = library_runtime_identity(self.capability, sources)

    def capability_profile(
        self, profile_id: str, execution_target: ExecutionTarget
    ) -> HardwareCapabilityProfile:
        """Convert one worker probe into the existing sanitized run profile."""

        if execution_target.target_class != self.sources.target_class:
            raise LibraryRuntimeError("library_execution_target_mismatch")
        capability = self.capability
        return capture_capability_profile(
            profile_id=profile_id,
            execution_target=execution_target,
            accelerator_backend=capability.accelerator_backend,
            accelerator_architecture=capability.accelerator_architecture,
            accelerator_model=capability.accelerator_model,
            accelerator_count=capability.accelerator_count,
            accelerator_memory_bytes=capability.accelerator_memory_bytes,
            system_memory_bytes=capability.system_memory_bytes,
            supported_precision_modes=capability.supported_precision_modes,
            supported_quantization_modes=capability.supported_quantization_modes,
            capabilities=capability.capabilities,
            library_versions=capability.library_versions,
        )

    def training_state_identity(
        self,
        experiment: Experiment,
        resolution: RecipeResolution,
        dataset_version: DatasetVersion,
        step: int,
    ) -> ContentIdentity:
        return library_training_state_identity(
            experiment,
            resolution,
            dataset_version,
            self.runtime_identity,
            step,
        )

    def validate_checkpoint(
        self, request: FixtureAdapterRequest, checkpoint: FixtureCheckpoint
    ) -> None:
        if not isinstance(request, FixtureAdapterRequest) or not isinstance(
            checkpoint, FixtureCheckpoint
        ):
            raise LibraryRuntimeError("library_checkpoint_invalid")
        expected_payload = ContentIdentity(
            "sha256", hashlib.sha256(checkpoint.payload).hexdigest()
        )
        expected_state = self.training_state_identity(
            request.experiment,
            request.recipe_resolution,
            request.dataset_version,
            checkpoint.step,
        )
        if (
            checkpoint.checkpoint_identity != expected_payload
            or checkpoint.training_state_identity != expected_state
            or checkpoint.step >= request.recipe_resolution.training_steps
            or checkpoint.resume_compatible is not True
        ):
            raise LibraryRuntimeError("library_checkpoint_invalid")

    def execute(
        self,
        request: FixtureAdapterRequest,
        *,
        control: FixtureControl | None = None,
        resume_checkpoint: FixtureCheckpoint | None = None,
    ) -> FixtureAdapterOutput:
        if not isinstance(request, FixtureAdapterRequest):
            raise LibraryRuntimeError("library_request_invalid")
        run_id = request.run.run_id
        with self._active_run_lock:
            if run_id in self._active_run_ids:
                raise LibraryRuntimeError("library_run_already_active")
            self._active_run_ids.add(run_id)
        try:
            return self._execute_owned(
                request,
                control=control,
                resume_checkpoint=resume_checkpoint,
            )
        finally:
            with self._active_run_lock:
                self._active_run_ids.discard(run_id)

    def _execute_owned(
        self,
        request: FixtureAdapterRequest,
        *,
        control: FixtureControl | None,
        resume_checkpoint: FixtureCheckpoint | None,
    ) -> FixtureAdapterOutput:
        self._validate_request(request, resume_checkpoint)
        active_control = control if control is not None else FixtureControl()
        if not isinstance(active_control, FixtureControl):
            raise LibraryRuntimeError("library_control_invalid")
        if active_control.cancel_after_step is not None and not (
            request.runtime_request.starting_step
            < active_control.cancel_after_step
            <= request.recipe_resolution.training_steps
        ):
            raise LibraryRuntimeError("library_control_out_of_range")
        if active_control.interrupt_after_step is not None and not (
            request.runtime_request.starting_step
            < active_control.interrupt_after_step
            < request.recipe_resolution.training_steps
        ):
            raise LibraryRuntimeError("library_control_out_of_range")
        input_payloads = {
            PortableLocation("inputs/rendered-dataset.jsonl"): request.rendered_dataset,
        }
        if resume_checkpoint is not None:
            input_payloads[PortableLocation("inputs/resume-checkpoint.bin")] = (
                resume_checkpoint.payload
            )
        input_members = {
            location: (
                "rendered_dataset"
                if location.logical_path.endswith("jsonl")
                else "resume_checkpoint",
                payload,
            )
            for location, payload in input_payloads.items()
        }
        input_manifest = build_transfer_manifest(
            TransferDirection.HOST_TO_WORKER, input_members
        )
        staging_base = self.sources.staging_root / request.run.run_id
        input_root = staging_base / "host-to-worker"
        input_receipt = stage_transfer(input_root, input_manifest, input_payloads)
        verified_inputs = read_verified_transfer(input_root, input_manifest)
        dataset_bytes = verified_inputs[
            PortableLocation("inputs/rendered-dataset.jsonl")
        ]
        resume_bytes = (
            verified_inputs[PortableLocation("inputs/resume-checkpoint.bin")]
            if resume_checkpoint is not None
            else None
        )
        controller = SerializedRunController(
            request.runtime_request.identity, request.run.run_id
        )
        messages: list[RuntimeMessage] = []
        sequence = 0

        def emit(kind: RuntimeMessageKind, payload: Mapping[str, object]) -> None:
            nonlocal sequence
            sequence += 1
            message = RuntimeMessage(
                request.runtime_request.identity,
                request.run.run_id,
                sequence,
                kind,
                payload,
            )
            controller.accept(message)
            messages.append(message)

        emit(
            RuntimeMessageKind.LAUNCHED,
            {
                "operation": RuntimeOperation.TRAIN.value,
                "target_class": self.sources.target_class,
            },
        )
        progress: list[FixtureProgress] = []
        checkpoints: list[FixtureCheckpoint] = []
        logs: list[FixtureLog] = [
            FixtureLog(
                1,
                "library_runtime_started",
                request.runtime_request.starting_step,
            )
        ]
        current_step = request.runtime_request.starting_step

        def on_progress(step: int, loss_microunits: int) -> None:
            nonlocal current_step
            current_step = step
            item = FixtureProgress(
                step,
                request.recipe_resolution.training_steps,
                loss_microunits,
            )
            progress.append(item)
            emit(
                RuntimeMessageKind.PROGRESS,
                {"step": step, "total_steps": item.total_steps},
            )
            emit(
                RuntimeMessageKind.METRIC,
                {
                    "name": "training_loss",
                    "value_microunits": loss_microunits,
                    "step": step,
                },
            )
            logs.append(FixtureLog(len(logs) + 1, "library_step_completed", step))
            if self._progress_callback is not None:
                self._progress_callback(step, item.total_steps)

        def on_checkpoint(item: LibraryCheckpointPayload) -> None:
            checkpoint = FixtureCheckpoint(
                item.step,
                self.training_state_identity(
                    request.experiment,
                    request.recipe_resolution,
                    request.dataset_version,
                    item.step,
                ),
                ContentIdentity("sha256", hashlib.sha256(item.payload).hexdigest()),
                item.payload,
                item.step < request.recipe_resolution.training_steps,
            )
            checkpoints.append(checkpoint)
            emit(RuntimeMessageKind.CHECKPOINT, checkpoint.to_receipt())
            logs.append(
                FixtureLog(len(logs) + 1, "library_checkpoint_saved", item.step)
            )

        def on_heartbeat(step: int) -> None:
            emit(
                RuntimeMessageKind.HEARTBEAT,
                {"step": step, "state": "training"},
            )

        def cancellation_requested() -> bool:
            return (
                active_control.cancel_after_step == current_step
                or self._cancellation_requested()
            )

        def interruption_requested() -> bool:
            return (
                active_control.interrupt_after_step == current_step
                or self._interruption_requested()
            )

        acquired = False
        try:
            self.resources.acquire(request.run.run_id, self.resource_names)
            acquired = True
            result = self.backend.train(
                context=LibraryExecutionContext(
                    request.runtime_request.identity,
                    request.run.run_id,
                    RuntimeOperation.TRAIN,
                    self.sources.target_class,
                ),
                model_source=self.sources.model_source,
                tokenizer_source=self.sources.tokenizer_source,
                rendered_dataset=dataset_bytes,
                resolution=request.recipe_resolution,
                resume_checkpoint=resume_bytes,
                on_progress=on_progress,
                on_checkpoint=on_checkpoint,
                on_heartbeat=on_heartbeat,
                cancellation_requested=cancellation_requested,
                interruption_requested=interruption_requested,
            )
        except Exception as exc:
            try:
                emit(
                    RuntimeMessageKind.FAILED,
                    {
                        "terminal": True,
                        "failure_code": "library_training_failed",
                        "phase": "training",
                    },
                )
            except RuntimeControllerError:
                pass
            raise LibraryAdapterExecutionError(
                getattr(exc, "code", "library_training_failed"),
                tuple(messages),
                (input_receipt,),
            ) from None
        finally:
            if acquired:
                self.resources.release(request.run.run_id)
        if result.cancelled:
            emit(
                RuntimeMessageKind.CANCELLATION_REQUESTED,
                {"acknowledged": True},
            )
            emit(RuntimeMessageKind.CANCELLED, {"terminal": True})
            logs.append(FixtureLog(len(logs) + 1, "library_cancelled", current_step))
            return FixtureAdapterOutput(
                FixtureTermination.CANCELLED,
                tuple(progress),
                tuple(checkpoints),
                tuple(logs),
                {},
                None,
                tuple(messages),
                (input_receipt, *result.transport_receipts),
            )
        if result.interrupted:
            if result.disconnected:
                emit(
                    RuntimeMessageKind.DISCONNECTED,
                    {"last_received_sequence": sequence},
                )
            emit(
                RuntimeMessageKind.INTERRUPTED,
                {
                    "terminal": True,
                    "recovery_checkpoint_count": len(checkpoints),
                },
            )
            logs.append(FixtureLog(len(logs) + 1, "library_interrupted", current_step))
            return FixtureAdapterOutput(
                FixtureTermination.INTERRUPTED,
                tuple(progress),
                tuple(checkpoints),
                tuple(logs),
                {},
                None,
                tuple(messages),
                (input_receipt, *result.transport_receipts),
            )
        if (
            result.adapter_payload is None
            or result.adapter_payload_format is None
            or result.adapter_config is None
        ):
            raise LibraryRuntimeError("library_artifact_output_missing")
        final_state = self.training_state_identity(
            request.experiment,
            request.recipe_resolution,
            request.dataset_version,
            request.recipe_resolution.training_steps,
        )
        try:
            members = self._artifact_members(
                request,
                result.adapter_payload,
                result.adapter_payload_format,
                result.adapter_config,
                final_state,
            )
            output_payloads = {
                PortableLocation(f"artifact/{path}"): payload
                for path, payload in members.items()
            }
            output_manifest = build_transfer_manifest(
                TransferDirection.WORKER_TO_HOST,
                {
                    location: ("artifact_member", payload)
                    for location, payload in output_payloads.items()
                },
            )
            output_root = staging_base / "worker-to-host"
            output_receipt = stage_transfer(
                output_root, output_manifest, output_payloads
            )
            verified_output = read_verified_transfer(output_root, output_manifest)
            members = {
                location.logical_path.removeprefix("artifact/"): payload
                for location, payload in verified_output.items()
            }
            bundle = build_bytes_bundle_manifest(members)
            emit(
                RuntimeMessageKind.ARTIFACT_READY,
                {
                    "bundle_identity": identity_fields(bundle.identity),
                    "member_count": len(bundle.members),
                },
            )
            emit(
                RuntimeMessageKind.COMPLETED,
                {"terminal": True, "verified_transfer": True},
            )
        except (LibraryRuntimeError, RuntimeControllerError, StagingError) as exc:
            try:
                emit(
                    RuntimeMessageKind.FAILED,
                    {
                        "terminal": True,
                        "failure_code": "library_artifact_transfer_failed",
                        "phase": "artifact_transfer",
                    },
                )
            except RuntimeControllerError:
                pass
            raise LibraryAdapterExecutionError(
                getattr(exc, "code", "library_artifact_transfer_failed"),
                tuple(messages),
                (input_receipt, *result.transport_receipts),
            ) from None
        logs.append(
            FixtureLog(
                len(logs) + 1,
                "library_artifact_emitted",
                request.recipe_resolution.training_steps,
            )
        )
        return FixtureAdapterOutput(
            FixtureTermination.COMPLETED,
            tuple(progress),
            tuple(checkpoints),
            tuple(logs),
            members,
            bundle,
            tuple(messages),
            (input_receipt, *result.transport_receipts, output_receipt),
        )

    def _validate_request(
        self,
        request: FixtureAdapterRequest,
        resume_checkpoint: FixtureCheckpoint | None,
    ) -> None:
        if not isinstance(request, FixtureAdapterRequest):
            raise LibraryRuntimeError("library_request_invalid")
        if (
            request.runtime_request.runtime_identity != self.runtime_identity
            or request.run.runtime_identity != self.runtime_identity
        ):
            raise LibraryRuntimeError("library_runtime_identity_mismatch")
        if (
            self.sources.base_model_revision != request.experiment.base_model_revision
            or self.sources.tokenizer_identity != request.experiment.tokenizer_identity
        ):
            raise LibraryRuntimeError("library_runtime_source_identity_mismatch")
        expected_state = self.training_state_identity(
            request.experiment,
            request.recipe_resolution,
            request.dataset_version,
            request.runtime_request.starting_step,
        )
        if request.runtime_request.training_state_identity != expected_state:
            raise LibraryRuntimeError("library_training_state_mismatch")
        required = {"accelerate", "peft", "torch", "transformers"}
        declared = request.recipe_resolution.library_versions
        observed = self.capability.library_versions
        if any(
            name not in declared
            or name not in observed
            or str(declared[name]) != observed[name]
            for name in required
        ):
            raise LibraryRuntimeError("library_version_mismatch")
        resuming = request.runtime_request.starting_step > 0
        if resuming != (resume_checkpoint is not None):
            raise LibraryRuntimeError("library_resume_checkpoint_missing")
        if resume_checkpoint is not None:
            self.validate_checkpoint(request, resume_checkpoint)
            if (
                request.runtime_request.resume_checkpoint_identity
                != resume_checkpoint.checkpoint_identity
            ):
                raise LibraryRuntimeError("library_resume_checkpoint_mismatch")

    def _artifact_members(
        self,
        request: FixtureAdapterRequest,
        adapter_payload: bytes,
        payload_format: str,
        library_adapter_config: Mapping[str, object],
        final_state: ContentIdentity,
    ) -> dict[str, bytes]:
        adapter_identity = ContentIdentity(
            "sha256", hashlib.sha256(adapter_payload).hexdigest()
        )
        config = dumps_canonical_json(
            {
                "schema_version": "v1",
                "adapter_type": request.recipe_resolution.adapter_type,
                "target_modules": list(request.recipe_resolution.target_modules),
                "rank": request.recipe_resolution.rank,
                "alpha": request.recipe_resolution.alpha,
                "base_model_revision": request.experiment.base_model_revision.to_dict(),
                "tokenizer_identity": identity_fields(
                    request.experiment.tokenizer_identity
                ),
                "compatibility_group": request.experiment.compatibility_group.to_dict(),
                "adapter_identity": identity_fields(adapter_identity),
                "runtime_identity": identity_fields(self.runtime_identity),
                "training_steps": request.recipe_resolution.training_steps,
                "runtime_kind": "library",
                "adapter_payload_format": payload_format,
                "library_versions": dict(
                    sorted(self.capability.library_versions.items())
                ),
                "library_adapter_config": dict(library_adapter_config),
            }
        )
        provenance = dumps_canonical_json(
            {
                "schema_version": "v1",
                "producing_run": record_reference(request.run).to_dict(),
                "resolved_runtime_request": record_reference(
                    request.runtime_request
                ).to_dict(),
                "experiment": record_reference(request.experiment).to_dict(),
                "experiment_manifest_identity": identity_fields(
                    request.experiment.manifest_identity
                ),
                "recipe_resolution": record_reference(
                    request.recipe_resolution
                ).to_dict(),
                "dataset_version_identity": identity_fields(
                    request.dataset_version.identity
                ),
                "final_training_state_identity": identity_fields(final_state),
            }
        )
        return {
            "adapter.bin": adapter_payload,
            "adapter_config.json": config,
            "provenance.json": provenance,
        }


class LibraryInferenceRuntime:
    """Load an already verified Temper-normalized library adapter for inference."""

    def __init__(
        self,
        backend: LibraryBackend,
        sources: LibraryRuntimeSources,
        runtime_identity: ContentIdentity,
    ) -> None:
        if (
            not isinstance(backend, LibraryBackend)
            or not isinstance(sources, LibraryRuntimeSources)
            or not isinstance(runtime_identity, ContentIdentity)
        ):
            raise LibraryRuntimeError("library_inference_runtime_invalid")
        self.backend = backend
        self.sources = sources
        capability = backend.probe()
        if not isinstance(capability, LibraryCapability):
            raise LibraryRuntimeError("library_capability_invalid")
        if library_runtime_identity(capability, sources) != runtime_identity:
            raise LibraryRuntimeError("library_inference_runtime_mismatch")
        self.capability = capability
        self.runtime_identity = runtime_identity

    def infer_verified(
        self,
        request: FixtureInferenceRequest,
        *,
        resolution: RecipeResolution,
        adapter_config: Mapping[str, Any],
        operation: RuntimeOperation = RuntimeOperation.INFER_FOCUSED,
    ) -> FixtureInferenceResult:
        if not isinstance(request, FixtureInferenceRequest) or not isinstance(
            resolution, RecipeResolution
        ):
            raise LibraryRuntimeError("library_inference_request_invalid")
        if adapter_config.get("runtime_kind") != "library":
            raise LibraryRuntimeError("library_inference_config_invalid")
        if adapter_config.get("runtime_identity") != identity_fields(
            self.runtime_identity
        ):
            raise LibraryRuntimeError("library_inference_runtime_mismatch")
        if adapter_config.get(
            "base_model_revision"
        ) != self.sources.base_model_revision.to_dict() or adapter_config.get(
            "tokenizer_identity"
        ) != identity_fields(self.sources.tokenizer_identity):
            raise LibraryRuntimeError("library_inference_source_mismatch")
        configured_versions = adapter_config.get("library_versions")
        if not isinstance(configured_versions, Mapping) or dict(
            configured_versions
        ) != dict(self.capability.library_versions):
            raise LibraryRuntimeError("library_inference_version_mismatch")
        payload_format = adapter_config.get("adapter_payload_format")
        if not isinstance(payload_format, str):
            raise LibraryRuntimeError("library_inference_config_invalid")
        if operation not in {
            RuntimeOperation.EVALUATE,
            RuntimeOperation.INFER_FOCUSED,
            RuntimeOperation.INFER_BATCH,
        }:
            raise LibraryRuntimeError("library_inference_operation_invalid")
        inputs = tuple(_inference_text(value) for value in request.inputs)
        request_identity = content_identity(
            LIBRARY_INFERENCE_REQUEST_PROJECTION,
            {
                "schema_version": "v1",
                "operation": operation.value,
                "runtime_identity": identity_fields(self.runtime_identity),
                "artifact_content_identity": identity_fields(
                    request.artifact_content_identity
                ),
                "settings": request.settings.to_dict(),
                "inputs": [thaw_json(value) for value in request.inputs],
            },
        )
        try:
            result = self.backend.infer(
                context=LibraryExecutionContext(
                    request_identity,
                    f"local-inference-{request_identity.value[:24]}",
                    operation,
                    self.sources.target_class,
                ),
                model_source=self.sources.model_source,
                tokenizer_source=self.sources.tokenizer_source,
                adapter_payload=request.adapter_bytes,
                adapter_payload_format=payload_format,
                resolution=resolution,
                settings=request.settings,
                inputs=inputs,
            )
        except LibraryRuntimeError:
            raise
        except Exception:
            raise LibraryRuntimeError("library_inference_failed") from None
        if not isinstance(result, LibraryInferenceResult):
            raise LibraryRuntimeError("library_inference_failed")
        if dict(result.library_versions) != dict(self.capability.library_versions):
            raise LibraryRuntimeError("library_inference_version_mismatch")
        frozen_outputs = tuple(
            freeze_json_object(
                {
                    "text": text,
                    "finish_reason": "library_complete",
                    "input_index": index,
                },
                field="library_inference_output",
            )
            for index, text in enumerate(result.outputs)
        )
        evidence = content_identity(
            LIBRARY_INFERENCE_EVIDENCE_PROJECTION,
            {
                "schema_version": "v1",
                "operation": operation.value,
                "request_identity": identity_fields(request_identity),
                "runtime_identity": identity_fields(self.runtime_identity),
                "artifact_content_identity": identity_fields(
                    request.artifact_content_identity
                ),
                "adapter_identity": identity_fields(
                    ContentIdentity(
                        "sha256", hashlib.sha256(request.adapter_bytes).hexdigest()
                    )
                ),
                "settings": request.settings.to_dict(),
                "input_count": len(request.inputs),
                "output_identities": [
                    identity_fields(
                        content_identity(
                            HashProjection("runtime.library_inference_output", "v1"),
                            thaw_json(output),
                        )
                    )
                    for output in frozen_outputs
                ],
                "library_versions": dict(sorted(result.library_versions.items())),
                "transfer_receipts": [
                    identity_fields(receipt.identity)
                    for receipt in result.transport_receipts
                ],
            },
        )
        return FixtureInferenceResult(
            request.inputs,
            frozen_outputs,
            request.settings,
            self.runtime_identity,
            evidence,
        )


def _inference_text(value: Mapping[str, Any]) -> str:
    thawed = thaw_json(value)
    for key in ("prompt", "text", "input"):
        item = thawed.get(key)
        if isinstance(item, str) and item:
            return item
    raise LibraryRuntimeError("library_inference_text_missing")
