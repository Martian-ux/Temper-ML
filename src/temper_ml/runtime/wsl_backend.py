"""Library backend implemented by the explicit Windows/WSL worker port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import uuid

from temper_ml.domain.projections import HashProjection, content_identity
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordValidationError,
    parse_identity,
    require_identifier,
)
from temper_ml.runtime.fixture_inference import InferenceSettings
from temper_ml.runtime.library_backend import (
    CancellationCheck,
    CheckpointCallback,
    HeartbeatCallback,
    InterruptionCheck,
    LibraryCapability,
    LibraryCheckpointPayload,
    LibraryExecutionContext,
    LibraryInferenceResult,
    LibraryRuntimeError,
    LibraryTrainingResult,
    ProgressCallback,
)
from temper_ml.runtime.paths import (
    PortableLocation,
    PortablePathError,
    WindowsWslPathMap,
)
from temper_ml.runtime.protocol import (
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperation,
)
from temper_ml.runtime.staging import (
    StagingError,
    TransferDirection,
    TransferManifest,
    TransferReceipt,
    build_transfer_manifest,
    read_verified_transfer,
    stage_transfer,
    verify_transfer,
)
from temper_ml.runtime.worker_port import (
    WorkerInvocation,
    WorkerLaunchResult,
    WorkerPortError,
    WorkerResponse,
    WslWorkerLauncher,
    WslWorkerLaunchSpec,
)
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)

WSL_PROBE_REQUEST_PROJECTION = HashProjection("runtime.wsl_probe_request", "v1")


@dataclass(frozen=True)
class WslWorkerConfig:
    """Private machine mapping for one explicitly selected WSL target."""

    target_class: str
    launch: WslWorkerLaunchSpec
    path_map: WindowsWslPathMap
    host_model_source: Path
    host_tokenizer_source: Path
    worker_model_source: PurePosixPath
    worker_tokenizer_source: PurePosixPath

    def __post_init__(self) -> None:
        try:
            require_identifier("target_class", self.target_class)
        except RecordValidationError:
            raise LibraryRuntimeError("wsl_worker_config_invalid") from None
        if not isinstance(self.launch, WslWorkerLaunchSpec) or not isinstance(
            self.path_map, WindowsWslPathMap
        ):
            raise LibraryRuntimeError("wsl_worker_config_invalid")
        for host_source in (self.host_model_source, self.host_tokenizer_source):
            if not isinstance(host_source, Path) or not host_source.is_absolute():
                raise LibraryRuntimeError("wsl_worker_config_invalid")
        for worker_source in (
            self.worker_model_source,
            self.worker_tokenizer_source,
        ):
            if (
                not isinstance(worker_source, PurePosixPath)
                or not worker_source.is_absolute()
            ):
                raise LibraryRuntimeError("wsl_worker_config_invalid")


class WslWorkerBackend:
    """Route the narrow library seam through one shared, verified WSL staging root."""

    def __init__(
        self,
        config: WslWorkerConfig,
        *,
        launcher: WslWorkerLauncher | None = None,
    ) -> None:
        if not isinstance(config, WslWorkerConfig):
            raise LibraryRuntimeError("wsl_worker_config_invalid")
        self.config = config
        self.launcher = launcher if launcher is not None else WslWorkerLauncher()
        if not isinstance(self.launcher, WslWorkerLauncher):
            raise LibraryRuntimeError("wsl_worker_launcher_invalid")
        self._capability: LibraryCapability | None = None
        self._probe_session = uuid.uuid4().hex

    def probe(self) -> LibraryCapability:
        if self._capability is not None:
            return self._capability
        request_identity = content_identity(
            WSL_PROBE_REQUEST_PROJECTION,
            {
                "schema_version": "v1",
                "target_class": self.config.target_class,
                "runtime_protocol": "v1",
            },
        )
        context = LibraryExecutionContext(
            request_identity,
            f"probe-{request_identity.value[:24]}",
            RuntimeOperation.PROBE,
            self.config.target_class,
        )
        prefix = PortableLocation(f"probes/{self._probe_session}")
        invocation = WorkerInvocation(
            context,
            self.config.path_map.worker_root,
            prefix,
        )
        result = self._launch(
            invocation,
            on_message=lambda message: None,
            cancellation_requested=lambda: False,
            interruption_requested=lambda: False,
        )
        if result.response.status != "completed":
            raise LibraryRuntimeError("wsl_probe_failed")
        outputs, receipt = self._verified_outputs(result.response)
        location = _response_location(
            result.response,
            "capability_location",
            "capability_profile",
            prefix,
        )
        try:
            value = loads_canonical_json(outputs[location])
        except (CanonicalJsonError, TypeError, ValueError):
            raise LibraryRuntimeError("wsl_probe_result_invalid") from None
        if (
            not isinstance(value, dict)
            or dumps_canonical_json(value) != outputs[location]
        ):
            raise LibraryRuntimeError("wsl_probe_result_invalid")
        try:
            capability = LibraryCapability(
                accelerator_backend=value["accelerator_backend"],
                accelerator_architecture=value["accelerator_architecture"],
                accelerator_model=value["accelerator_model"],
                accelerator_count=value["accelerator_count"],
                accelerator_memory_bytes=tuple(value["accelerator_memory_bytes"]),
                system_memory_bytes=value["system_memory_bytes"],
                supported_precision_modes=tuple(value["supported_precision_modes"]),
                supported_quantization_modes=tuple(
                    value["supported_quantization_modes"]
                ),
                capabilities=tuple(value["capabilities"]),
                library_versions=value["library_versions"],
            )
        except (KeyError, TypeError, ValueError):
            raise LibraryRuntimeError("wsl_probe_result_invalid") from None
        del receipt
        self._capability = capability
        return capability

    def train(
        self,
        *,
        context: LibraryExecutionContext,
        model_source: Path,
        tokenizer_source: Path,
        rendered_dataset: bytes,
        resolution: RecipeResolution,
        resume_checkpoint: bytes | None,
        on_progress: ProgressCallback,
        on_checkpoint: CheckpointCallback,
        on_heartbeat: HeartbeatCallback,
        cancellation_requested: CancellationCheck,
        interruption_requested: InterruptionCheck,
    ) -> LibraryTrainingResult:
        self._validate_sources(context, model_source, tokenizer_source)
        if context.operation is not RuntimeOperation.TRAIN:
            raise LibraryRuntimeError("wsl_training_context_invalid")
        frozen_capability = self.probe()
        prefix = self._operation_prefix(context)
        dataset_location = PortableLocation(
            f"{prefix.logical_path}/inputs/rendered-dataset.jsonl"
        )
        payloads = {dataset_location: rendered_dataset}
        members = {dataset_location: ("rendered_dataset", rendered_dataset)}
        if resume_checkpoint is not None:
            checkpoint_location = PortableLocation(
                f"{prefix.logical_path}/inputs/resume-checkpoint.bin"
            )
            payloads[checkpoint_location] = resume_checkpoint
            members[checkpoint_location] = (
                "resume_checkpoint",
                resume_checkpoint,
            )
        input_manifest = build_transfer_manifest(
            TransferDirection.HOST_TO_WORKER, members
        )
        input_receipt = stage_transfer(self._host_root(), input_manifest, payloads)
        invocation = WorkerInvocation(
            context,
            self.config.path_map.worker_root,
            prefix,
            self.config.worker_model_source,
            self.config.worker_tokenizer_source,
            resolution,
            input_manifest,
        )
        progress: list[tuple[int, int]] = []
        checkpoints: dict[int, LibraryCheckpointPayload] = {}
        pending_progress: dict[int, int] = {}
        checkpoint_receipts: list[TransferReceipt] = []

        def handle(message: RuntimeMessage) -> None:
            if message.kind is RuntimeMessageKind.PROGRESS:
                pending_progress[int(message.payload["step"])] = int(
                    message.payload["total_steps"]
                )
            elif message.kind is RuntimeMessageKind.METRIC:
                step = int(message.payload["step"])
                if step not in pending_progress:
                    raise WorkerPortError("wsl_progress_order_invalid")
                loss = int(message.payload["value_microunits"])
                if not progress or progress[-1][0] != step:
                    progress.append((step, loss))
                    on_progress(step, loss)
            elif message.kind is RuntimeMessageKind.CHECKPOINT:
                step = int(message.payload["step"])
                if step in checkpoints:
                    return
                location = self._checkpoint_location(prefix, step)
                manifest = TransferManifest(
                    TransferDirection.WORKER_TO_HOST,
                    (
                        _member_from_message(
                            location, "resume_checkpoint", message.payload
                        ),
                    ),
                )
                receipt = verify_transfer(self._host_root(), manifest)
                payload = read_verified_transfer(self._host_root(), manifest)[location]
                item = LibraryCheckpointPayload(step, payload)
                checkpoints[step] = item
                checkpoint_receipts.append(receipt)
                on_checkpoint(item)
            elif message.kind is RuntimeMessageKind.HEARTBEAT:
                on_heartbeat(int(message.payload["step"]))

        try:
            launch_result = self._launch(
                invocation,
                on_message=handle,
                cancellation_requested=cancellation_requested,
                interruption_requested=interruption_requested,
            )
        except WorkerPortError as exc:
            if exc.code in {
                "worker_disconnected",
                "worker_timeout",
                "worker_reconciliation_required",
            }:
                if exc.code == "worker_reconciliation_required":
                    try:
                        for message in exc.messages:
                            handle(message)
                    except WorkerPortError as replay_error:
                        raise LibraryRuntimeError(replay_error.code) from None
                return LibraryTrainingResult(
                    None,
                    None,
                    None,
                    tuple(progress),
                    tuple(checkpoints[step] for step in sorted(checkpoints)),
                    interrupted=True,
                    disconnected=True,
                    transport_receipts=(
                        input_receipt,
                        *checkpoint_receipts,
                    ),
                )
            raise LibraryRuntimeError(exc.code) from None
        response = launch_result.response
        if response.status == "failed":
            raise LibraryRuntimeError("wsl_worker_operation_failed")
        if response.status == "cancelled":
            return LibraryTrainingResult(
                None,
                None,
                None,
                tuple(progress),
                tuple(checkpoints[step] for step in sorted(checkpoints)),
                cancelled=True,
                transport_receipts=(input_receipt, *checkpoint_receipts),
            )
        if response.status == "interrupted":
            receipts: tuple[TransferReceipt, ...] = ()
            if response.output_manifest is not None:
                _, output_receipt = self._verified_outputs(response)
                receipts = (output_receipt,)
            return LibraryTrainingResult(
                None,
                None,
                None,
                tuple(progress),
                tuple(checkpoints[step] for step in sorted(checkpoints)),
                interrupted=True,
                transport_receipts=(
                    input_receipt,
                    *checkpoint_receipts,
                    *receipts,
                ),
            )
        outputs, output_receipt = self._verified_outputs(response)
        adapter_location = _response_location(
            response, "adapter_location", "adapter_payload", prefix
        )
        metadata_location = _response_location(
            response, "metadata_location", "training_result", prefix
        )
        metadata = _canonical_object(
            outputs[metadata_location], "wsl_training_result_invalid"
        )
        if (
            set(metadata)
            != {
                "schema_version",
                "adapter_payload_format",
                "adapter_config",
                "library_versions",
            }
            or metadata["schema_version"] != "v1"
            or not isinstance(metadata["library_versions"], Mapping)
        ):
            raise LibraryRuntimeError("wsl_training_result_invalid")
        if dict(metadata["library_versions"]) != dict(
            frozen_capability.library_versions
        ):
            raise LibraryRuntimeError("wsl_training_library_versions_mismatch")
        adapter_config = metadata["adapter_config"]
        if not isinstance(adapter_config, Mapping):
            raise LibraryRuntimeError("wsl_training_result_invalid")
        return LibraryTrainingResult(
            outputs[adapter_location],
            metadata["adapter_payload_format"],
            adapter_config,
            tuple(progress),
            tuple(checkpoints[step] for step in sorted(checkpoints)),
            transport_receipts=(
                input_receipt,
                *checkpoint_receipts,
                output_receipt,
            ),
        )

    def infer(
        self,
        *,
        context: LibraryExecutionContext,
        model_source: Path,
        tokenizer_source: Path,
        adapter_payload: bytes,
        adapter_payload_format: str,
        resolution: RecipeResolution,
        settings: InferenceSettings,
        inputs: tuple[str, ...],
    ) -> LibraryInferenceResult:
        self._validate_sources(context, model_source, tokenizer_source)
        if context.operation not in {
            RuntimeOperation.EVALUATE,
            RuntimeOperation.INFER_FOCUSED,
            RuntimeOperation.INFER_BATCH,
        }:
            raise LibraryRuntimeError("wsl_inference_context_invalid")
        prefix = self._operation_prefix(context)
        adapter_location = PortableLocation(f"{prefix.logical_path}/inputs/adapter.bin")
        inputs_location = PortableLocation(
            f"{prefix.logical_path}/inputs/inference.json"
        )
        input_bytes = dumps_canonical_json(list(inputs))
        payloads = {
            adapter_location: adapter_payload,
            inputs_location: input_bytes,
        }
        manifest = build_transfer_manifest(
            TransferDirection.HOST_TO_WORKER,
            {
                adapter_location: ("adapter_payload", adapter_payload),
                inputs_location: ("inference_inputs", input_bytes),
            },
        )
        input_receipt = stage_transfer(self._host_root(), manifest, payloads)
        invocation = WorkerInvocation(
            context,
            self.config.path_map.worker_root,
            prefix,
            self.config.worker_model_source,
            self.config.worker_tokenizer_source,
            resolution,
            manifest,
            settings,
            adapter_payload_format,
        )
        try:
            result = self._launch(
                invocation,
                on_message=lambda message: None,
                cancellation_requested=lambda: False,
                interruption_requested=lambda: False,
            )
        except WorkerPortError as exc:
            raise LibraryRuntimeError(exc.code) from None
        if result.response.status != "completed":
            raise LibraryRuntimeError("wsl_inference_failed")
        outputs, output_receipt = self._verified_outputs(result.response)
        location = _response_location(
            result.response,
            "inference_location",
            "inference_result",
            prefix,
        )
        value = _canonical_object(outputs[location], "wsl_inference_result_invalid")
        if (
            set(value) != {"schema_version", "outputs", "library_versions"}
            or value["schema_version"] != "v1"
            or not isinstance(value["outputs"], list)
            or any(not isinstance(item, str) for item in value["outputs"])
            or len(value["outputs"]) != len(inputs)
            or not isinstance(value["library_versions"], Mapping)
        ):
            raise LibraryRuntimeError("wsl_inference_result_invalid")
        return LibraryInferenceResult(
            tuple(value["outputs"]),
            value["library_versions"],
            (input_receipt, output_receipt),
        )

    def _launch(
        self,
        invocation: WorkerInvocation,
        *,
        on_message: Any,
        cancellation_requested: CancellationCheck,
        interruption_requested: InterruptionCheck,
    ) -> WorkerLaunchResult:
        prefix = invocation.output_prefix.logical_path
        invocation_location = PortableLocation(f"{prefix}/invocation.json")
        return self.launcher.launch(
            self.config.launch,
            invocation,
            invocation_host_path=self._host_path(invocation_location),
            invocation_worker_path=self.config.path_map.worker_path(
                invocation_location
            ),
            message_host_root=self._host_path(invocation.message_prefix),
            response_host_path=self._host_path(invocation.response_location),
            cancellation_host_path=self._host_path(invocation.cancellation_location),
            interruption_host_path=self._host_path(invocation.interruption_location),
            on_message=on_message,
            cancellation_requested=cancellation_requested,
            interruption_requested=interruption_requested,
        )

    def _verified_outputs(
        self, response: WorkerResponse
    ) -> tuple[dict[PortableLocation, bytes], TransferReceipt]:
        if response.output_manifest is None:
            raise LibraryRuntimeError("wsl_output_manifest_missing")
        try:
            receipt = verify_transfer(self._host_root(), response.output_manifest)
            outputs = read_verified_transfer(
                self._host_root(), response.output_manifest
            )
        except StagingError:
            raise LibraryRuntimeError("wsl_output_transfer_invalid") from None
        return outputs, receipt

    def _validate_sources(
        self,
        context: LibraryExecutionContext,
        model_source: Path,
        tokenizer_source: Path,
    ) -> None:
        if (
            not isinstance(context, LibraryExecutionContext)
            or context.target_class != self.config.target_class
            or model_source != self.config.host_model_source
            or tokenizer_source != self.config.host_tokenizer_source
        ):
            raise LibraryRuntimeError("wsl_worker_source_mismatch")

    @staticmethod
    def _operation_prefix(context: LibraryExecutionContext) -> PortableLocation:
        return PortableLocation(
            "operations/"
            f"{context.run_id}-{context.request_identity.value[:16]}-"
            f"{context.operation.value}"
        )

    def _checkpoint_location(
        self, prefix: PortableLocation, step: int
    ) -> PortableLocation:
        return PortableLocation(
            f"{prefix.logical_path}/outputs/checkpoints/{step:08d}.bin"
        )

    def _host_root(self) -> Path:
        return Path(str(self.config.path_map.host_root))

    def _host_path(self, location: PortableLocation) -> Path:
        try:
            return Path(str(self.config.path_map.host_path(location)))
        except PortablePathError:
            raise LibraryRuntimeError("wsl_staging_path_invalid") from None


def _member_from_message(
    location: PortableLocation,
    role: str,
    payload: Mapping[str, Any],
):
    from temper_ml.runtime.staging import TransferMember

    try:
        identity = parse_identity(
            payload["checkpoint_identity"], field="checkpoint_identity"
        )
        byte_count = payload["byte_count"]
        return TransferMember(location, role, identity, byte_count)
    except (KeyError, RecordValidationError, TypeError, ValueError):
        raise WorkerPortError("wsl_checkpoint_message_invalid") from None


def _response_location(
    response: WorkerResponse,
    field: str,
    expected_role: str,
    expected_prefix: PortableLocation,
) -> PortableLocation:
    raw = response.payload.get(field)
    if not isinstance(raw, Mapping) or set(raw) != {"logical_path"}:
        raise LibraryRuntimeError("wsl_response_location_invalid")
    try:
        location = PortableLocation(raw["logical_path"])
    except (PortablePathError, TypeError, ValueError):
        raise LibraryRuntimeError("wsl_response_location_invalid") from None
    if not location.logical_path.startswith(f"{expected_prefix.logical_path}/outputs/"):
        raise LibraryRuntimeError("wsl_response_location_invalid")
    if response.output_manifest is None:
        raise LibraryRuntimeError("wsl_output_manifest_missing")
    matches = tuple(
        member
        for member in response.output_manifest.members
        if member.logical_location == location and member.role == expected_role
    )
    if len(matches) != 1:
        raise LibraryRuntimeError("wsl_response_location_invalid")
    return location


def _canonical_object(data: bytes, code: str) -> dict[str, Any]:
    try:
        value = loads_canonical_json(data)
    except (CanonicalJsonError, TypeError, ValueError):
        raise LibraryRuntimeError(code) from None
    if not isinstance(value, dict) or dumps_canonical_json(value) != data:
        raise LibraryRuntimeError(code)
    return value
