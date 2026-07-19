"""WSL worker entrypoint; it owns staged bytes but never canonical records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from temper_ml.domain.projections import HashProjection, content_identity
from temper_ml.domain.records import identity_fields, thaw_json
from temper_ml.runtime.library_backend import (
    LibraryCheckpointPayload,
    LibraryRuntimeError,
    LibraryTrainingResult,
    TransformersPeftBackend,
)
from temper_ml.runtime.paths import PortableLocation
from temper_ml.runtime.protocol import (
    RuntimeMessageKind,
    RuntimeOperation,
)
from temper_ml.runtime.staging import (
    StagingError,
    TransferDirection,
    TransferManifest,
    build_transfer_manifest,
    read_verified_transfer,
    stage_transfer,
)
from temper_ml.runtime.worker_port import (
    WorkerEventSink,
    WorkerInvocation,
    WorkerPortError,
    WorkerResponse,
    control_requested,
    write_worker_response,
)
from temper_ml.store.canonical_json import (
    dumps_canonical_json,
    loads_canonical_json,
)
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes

WORKER_CHECKPOINT_STATE_PROJECTION = HashProjection(
    "runtime.worker_checkpoint_state", "v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="temper-runtime-worker")
    parser.add_argument("--request", required=True)
    arguments = parser.parse_args(argv)
    path = Path(arguments.request)
    try:
        if not path.is_absolute():
            raise WorkerPortError("worker_invocation_path_invalid")
        invocation = WorkerInvocation.from_private_bytes(read_stable_bytes(path))
        return _execute(invocation)
    except (SafeIoError, WorkerPortError):
        return 2


def _execute(invocation: WorkerInvocation) -> int:
    sink = WorkerEventSink(invocation)
    sink.emit(
        RuntimeMessageKind.LAUNCHED,
        {
            "operation": invocation.context.operation.value,
            "target_class": invocation.context.target_class,
        },
    )
    try:
        backend = TransformersPeftBackend()
        if invocation.context.operation is RuntimeOperation.PROBE:
            _probe(invocation, sink, backend)
        elif invocation.context.operation is RuntimeOperation.TRAIN:
            _train(invocation, sink, backend)
        else:
            _infer(invocation, sink, backend)
        return 0
    except (LibraryRuntimeError, StagingError, WorkerPortError) as exc:
        _record_failure(invocation, sink, exc.code)
        return 0
    except Exception:
        _record_failure(invocation, sink, "worker_internal_failure")
        return 0


def _probe(
    invocation: WorkerInvocation,
    sink: WorkerEventSink,
    backend: TransformersPeftBackend,
) -> None:
    capability = backend.probe()
    location = _output_location(invocation, "capability.json")
    payloads = {location: dumps_canonical_json(capability.to_public_facts())}
    manifest = _stage_outputs(
        invocation, {location: ("capability_profile", payloads[location])}
    )
    response = WorkerResponse(
        invocation.context,
        "completed",
        manifest,
        {"capability_location": location.to_dict()},
    )
    write_worker_response(invocation, response)
    _complete(sink, manifest)


def _train(
    invocation: WorkerInvocation,
    sink: WorkerEventSink,
    backend: TransformersPeftBackend,
) -> None:
    if (
        invocation.input_manifest is None
        or invocation.resolution is None
        or invocation.model_source is None
        or invocation.tokenizer_source is None
    ):
        raise WorkerPortError("worker_invocation_invalid")
    resolution = invocation.resolution
    inputs = read_verified_transfer(
        Path(invocation.worker_root.as_posix()), invocation.input_manifest
    )
    rendered_dataset = _one_input(inputs, invocation.input_manifest, "rendered_dataset")
    resume_checkpoint = _optional_input(
        inputs, invocation.input_manifest, "resume_checkpoint"
    )
    checkpoints: list[LibraryCheckpointPayload] = []
    cancellation_acknowledged = False

    def on_progress(step: int, loss_microunits: int) -> None:
        sink.emit(
            RuntimeMessageKind.PROGRESS,
            {
                "step": step,
                "total_steps": resolution.training_steps,
            },
        )
        sink.emit(
            RuntimeMessageKind.METRIC,
            {
                "name": "training_loss",
                "value_microunits": loss_microunits,
                "step": step,
            },
        )

    def on_checkpoint(checkpoint: LibraryCheckpointPayload) -> None:
        checkpoints.append(checkpoint)
        location = _checkpoint_location(invocation, checkpoint.step)
        stage_transfer(
            Path(invocation.worker_root.as_posix()),
            build_transfer_manifest(
                TransferDirection.WORKER_TO_HOST,
                {location: ("resume_checkpoint", checkpoint.payload)},
            ),
            {location: checkpoint.payload},
        )
        checkpoint_identity = content_identity(
            WORKER_CHECKPOINT_STATE_PROJECTION,
            {
                "schema_version": "v1",
                "request_identity": identity_fields(
                    invocation.context.request_identity
                ),
                "step": checkpoint.step,
                "checkpoint_sha256": _sha256(checkpoint.payload),
            },
        )
        sink.emit(
            RuntimeMessageKind.CHECKPOINT,
            {
                "step": checkpoint.step,
                "training_state_identity": identity_fields(checkpoint_identity),
                "checkpoint_identity": {
                    "algorithm": "sha256",
                    "value": _sha256(checkpoint.payload),
                },
                "byte_count": len(checkpoint.payload),
                "resume_compatible": checkpoint.step < resolution.training_steps,
            },
        )

    def on_heartbeat(step: int) -> None:
        sink.emit(RuntimeMessageKind.HEARTBEAT, {"step": step, "state": "training"})

    def cancellation_requested() -> bool:
        nonlocal cancellation_acknowledged
        requested = control_requested(invocation, "cancel")
        if requested and not cancellation_acknowledged:
            sink.emit(
                RuntimeMessageKind.CANCELLATION_REQUESTED,
                {"acknowledged": True},
            )
            cancellation_acknowledged = True
        return requested

    result = backend.train(
        context=invocation.context,
        model_source=Path(invocation.model_source.as_posix()),
        tokenizer_source=Path(invocation.tokenizer_source.as_posix()),
        rendered_dataset=rendered_dataset,
        resolution=resolution,
        resume_checkpoint=resume_checkpoint,
        on_progress=on_progress,
        on_checkpoint=on_checkpoint,
        on_heartbeat=on_heartbeat,
        cancellation_requested=cancellation_requested,
        interruption_requested=lambda: control_requested(invocation, "interrupt"),
    )
    if result.cancelled:
        response = WorkerResponse(invocation.context, "cancelled", None, {})
        write_worker_response(invocation, response)
        sink.emit(RuntimeMessageKind.CANCELLED, {"terminal": True})
        return
    if result.interrupted:
        manifest = _checkpoint_manifest(invocation, checkpoints)
        response = WorkerResponse(
            invocation.context,
            "interrupted",
            manifest,
            {"checkpoints": _checkpoint_descriptors(invocation, checkpoints)},
        )
        write_worker_response(invocation, response)
        sink.emit(
            RuntimeMessageKind.INTERRUPTED,
            {
                "terminal": True,
                "recovery_checkpoint_count": len(checkpoints),
            },
        )
        return
    _complete_training(invocation, sink, backend, result, checkpoints)


def _complete_training(
    invocation: WorkerInvocation,
    sink: WorkerEventSink,
    backend: TransformersPeftBackend,
    result: LibraryTrainingResult,
    checkpoints: list[LibraryCheckpointPayload],
) -> None:
    if (
        result.adapter_payload is None
        or result.adapter_payload_format is None
        or result.adapter_config is None
    ):
        raise WorkerPortError("worker_training_output_invalid")
    capability = backend.probe()
    adapter_location = _output_location(invocation, "adapter.bin")
    metadata_location = _output_location(invocation, "training-result.json")
    metadata = dumps_canonical_json(
        {
            "schema_version": "v1",
            "adapter_payload_format": result.adapter_payload_format,
            "adapter_config": thaw_json(result.adapter_config),
            "library_versions": dict(capability.library_versions),
        }
    )
    payloads = {
        adapter_location: result.adapter_payload,
        metadata_location: metadata,
    }
    members = {
        adapter_location: ("adapter_payload", result.adapter_payload),
        metadata_location: ("training_result", metadata),
    }
    for checkpoint in checkpoints:
        location = _checkpoint_location(invocation, checkpoint.step)
        payloads[location] = checkpoint.payload
        members[location] = ("resume_checkpoint", checkpoint.payload)
    manifest = build_transfer_manifest(TransferDirection.WORKER_TO_HOST, members)
    stage_transfer(Path(invocation.worker_root.as_posix()), manifest, payloads)
    response = WorkerResponse(
        invocation.context,
        "completed",
        manifest,
        {
            "adapter_location": adapter_location.to_dict(),
            "metadata_location": metadata_location.to_dict(),
            "checkpoints": _checkpoint_descriptors(invocation, checkpoints),
        },
    )
    write_worker_response(invocation, response)
    _complete(sink, manifest)


def _infer(
    invocation: WorkerInvocation,
    sink: WorkerEventSink,
    backend: TransformersPeftBackend,
) -> None:
    if (
        invocation.input_manifest is None
        or invocation.resolution is None
        or invocation.model_source is None
        or invocation.tokenizer_source is None
        or invocation.settings is None
        or invocation.adapter_payload_format is None
    ):
        raise WorkerPortError("worker_invocation_invalid")
    inputs = read_verified_transfer(
        Path(invocation.worker_root.as_posix()), invocation.input_manifest
    )
    adapter = _one_input(inputs, invocation.input_manifest, "adapter_payload")
    raw_inputs = _one_input(inputs, invocation.input_manifest, "inference_inputs")
    try:
        decoded = loads_canonical_json(raw_inputs)
    except (TypeError, ValueError):
        raise WorkerPortError("worker_inference_inputs_invalid") from None
    if (
        not isinstance(decoded, list)
        or not decoded
        or any(not isinstance(value, str) or not value for value in decoded)
    ):
        raise WorkerPortError("worker_inference_inputs_invalid")
    result = backend.infer(
        context=invocation.context,
        model_source=Path(invocation.model_source.as_posix()),
        tokenizer_source=Path(invocation.tokenizer_source.as_posix()),
        adapter_payload=adapter,
        adapter_payload_format=invocation.adapter_payload_format,
        resolution=invocation.resolution,
        settings=invocation.settings,
        inputs=tuple(decoded),
    )
    location = _output_location(invocation, "inference.json")
    payload = dumps_canonical_json(
        {
            "schema_version": "v1",
            "outputs": list(result.outputs),
            "library_versions": dict(result.library_versions),
        }
    )
    manifest = _stage_outputs(invocation, {location: ("inference_result", payload)})
    response = WorkerResponse(
        invocation.context,
        "completed",
        manifest,
        {"inference_location": location.to_dict()},
    )
    write_worker_response(invocation, response)
    _complete(sink, manifest)


def _complete(sink: WorkerEventSink, manifest: TransferManifest) -> None:
    sink.emit(
        RuntimeMessageKind.ARTIFACT_READY,
        {
            "bundle_identity": {
                "algorithm": manifest.identity.algorithm,
                "value": manifest.identity.value,
            },
            "member_count": len(manifest.members),
        },
    )
    sink.emit(
        RuntimeMessageKind.COMPLETED,
        {"terminal": True, "verified_transfer": True},
    )


def _record_failure(
    invocation: WorkerInvocation, sink: WorkerEventSink, code: str
) -> None:
    try:
        response = WorkerResponse(
            invocation.context,
            "failed",
            None,
            {"failure_code": _safe_failure_code(code)},
        )
        write_worker_response(invocation, response)
        sink.emit(
            RuntimeMessageKind.FAILED,
            {
                "terminal": True,
                "failure_code": _safe_failure_code(code),
                "phase": "worker_operation",
            },
        )
    except Exception:
        return


def _stage_outputs(
    invocation: WorkerInvocation,
    members: Mapping[PortableLocation, tuple[str, bytes]],
) -> TransferManifest:
    manifest = build_transfer_manifest(TransferDirection.WORKER_TO_HOST, members)
    stage_transfer(
        Path(invocation.worker_root.as_posix()),
        manifest,
        {location: value[1] for location, value in members.items()},
    )
    return manifest


def _checkpoint_manifest(
    invocation: WorkerInvocation,
    checkpoints: list[LibraryCheckpointPayload],
) -> TransferManifest | None:
    if not checkpoints:
        return None
    return build_transfer_manifest(
        TransferDirection.WORKER_TO_HOST,
        {
            _checkpoint_location(invocation, checkpoint.step): (
                "resume_checkpoint",
                checkpoint.payload,
            )
            for checkpoint in checkpoints
        },
    )


def _checkpoint_descriptors(
    invocation: WorkerInvocation,
    checkpoints: list[LibraryCheckpointPayload],
) -> list[dict[str, object]]:
    return [
        {
            "step": checkpoint.step,
            "location": _checkpoint_location(invocation, checkpoint.step).to_dict(),
        }
        for checkpoint in checkpoints
    ]


def _one_input(
    values: Mapping[PortableLocation, bytes],
    manifest: TransferManifest,
    role: str,
) -> bytes:
    locations = tuple(
        member.logical_location for member in manifest.members if member.role == role
    )
    if len(locations) != 1:
        raise WorkerPortError("worker_input_manifest_invalid")
    return values[locations[0]]


def _optional_input(
    values: Mapping[PortableLocation, bytes],
    manifest: TransferManifest,
    role: str,
) -> bytes | None:
    locations = tuple(
        member.logical_location for member in manifest.members if member.role == role
    )
    if len(locations) > 1:
        raise WorkerPortError("worker_input_manifest_invalid")
    return values[locations[0]] if locations else None


def _output_location(invocation: WorkerInvocation, filename: str) -> PortableLocation:
    return PortableLocation(
        f"{invocation.output_prefix.logical_path}/outputs/{filename}"
    )


def _checkpoint_location(invocation: WorkerInvocation, step: int) -> PortableLocation:
    return PortableLocation(
        f"{invocation.output_prefix.logical_path}/outputs/checkpoints/{step:08d}.bin"
    )


def _safe_failure_code(code: str) -> str:
    if not isinstance(code, str) or not code or len(code) > 128:
        return "worker_operation_failed"
    return (
        code
        if all(character.isalnum() or character == "_" for character in code)
        else "worker_operation_failed"
    )


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
