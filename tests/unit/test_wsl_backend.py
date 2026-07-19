from decimal import Decimal
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import RecordReference, identity_fields
from temper_ml.runtime.library_backend import (
    LibraryExecutionContext,
    LibraryRuntimeError,
)
from temper_ml.runtime.paths import PortableLocation, WindowsWslPathMap
from temper_ml.runtime.protocol import (
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperation,
)
from temper_ml.runtime.staging import (
    TransferDirection,
    build_transfer_manifest,
    stage_transfer,
)
from temper_ml.runtime.worker_port import (
    WorkerLaunchResult,
    WorkerPortError,
    WorkerResponse,
    WslWorkerLauncher,
    WslWorkerLaunchSpec,
)
from temper_ml.runtime.wsl_backend import WslWorkerBackend, WslWorkerConfig


IDENTITY = ContentIdentity("sha256", "4" * 64)


def _reference(record_type: str, logical_id: str) -> RecordReference:
    digest = hashlib.sha256(f"{record_type}:{logical_id}".encode()).hexdigest()
    return RecordReference(
        record_type,
        logical_id,
        ContentIdentity("sha256", digest),
    )


def _resolution() -> RecipeResolution:
    return RecipeResolution(
        resolution_id="resolution-wsl-test",
        recipe=_reference("recipe", "recipe-wsl-test"),
        base_model_revision=_reference("base_model_revision", "model-wsl-test"),
        hardware_requirements=_reference(
            "hardware_requirements", "requirements-wsl-test"
        ),
        execution_target=_reference("execution_target", "target-wsl-test"),
        adapter_type="lora",
        target_modules=("q_proj",),
        rank=2,
        alpha=4,
        dropout=0,
        learning_rate=Decimal("0.0002"),
        effective_batch_size=1,
        sequence_length=32,
        optimizer="adamw",
        precision="fp32",
        gradient_accumulation=1,
        seed=7,
        schedule="linear",
        training_steps=2,
        checkpoint_cadence=1,
        quantization="none",
        library_versions={
            "accelerate": "1.test",
            "peft": "1.test",
            "torch": "1.test",
            "transformers": "1.test",
        },
        applied_constraints=(),
    )


def _config(root: Path) -> WslWorkerConfig:
    return WslWorkerConfig(
        target_class="wsl_rocm",
        launch=WslWorkerLaunchSpec("Ubuntu-ROCm", PurePosixPath("/usr/bin/python3")),
        path_map=WindowsWslPathMap(
            PureWindowsPath(str(root)), PurePosixPath("/temper-staging")
        ),
        host_model_source=(root / "model").resolve(),
        host_tokenizer_source=(root / "tokenizer").resolve(),
        worker_model_source=PurePosixPath("/models/base"),
        worker_tokenizer_source=PurePosixPath("/models/tokenizer"),
    )


def test_partial_worker_ledger_replays_verified_checkpoint_without_relaunch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint = b"synthetic-checkpoint"
    observed_progress: list[tuple[int, int]] = []
    observed_checkpoints = []
    observed_heartbeats: list[int] = []

    class PartialLauncher(WslWorkerLauncher):
        def launch(self, spec, invocation, **kwargs):
            del spec, kwargs
            location = PortableLocation(
                f"{invocation.output_prefix.logical_path}/outputs/"
                "checkpoints/00000001.bin"
            )
            manifest = build_transfer_manifest(
                TransferDirection.WORKER_TO_HOST,
                {location: ("resume_checkpoint", checkpoint)},
            )
            stage_transfer(tmp_path, manifest, {location: checkpoint})
            messages = (
                RuntimeMessage(
                    invocation.context.request_identity,
                    invocation.context.run_id,
                    1,
                    RuntimeMessageKind.LAUNCHED,
                    {"operation": "train", "target_class": "wsl_rocm"},
                ),
                RuntimeMessage(
                    invocation.context.request_identity,
                    invocation.context.run_id,
                    2,
                    RuntimeMessageKind.PROGRESS,
                    {"step": 1, "total_steps": 2},
                ),
                RuntimeMessage(
                    invocation.context.request_identity,
                    invocation.context.run_id,
                    3,
                    RuntimeMessageKind.METRIC,
                    {
                        "name": "training_loss",
                        "value_microunits": 23,
                        "step": 1,
                    },
                ),
                RuntimeMessage(
                    invocation.context.request_identity,
                    invocation.context.run_id,
                    4,
                    RuntimeMessageKind.CHECKPOINT,
                    {
                        "step": 1,
                        "checkpoint_identity": identity_fields(
                            manifest.members[0].content_identity
                        ),
                        "training_state_identity": identity_fields(IDENTITY),
                        "byte_count": len(checkpoint),
                        "resume_compatible": True,
                    },
                ),
                RuntimeMessage(
                    invocation.context.request_identity,
                    invocation.context.run_id,
                    5,
                    RuntimeMessageKind.HEARTBEAT,
                    {"step": 1, "state": "training"},
                ),
            )
            raise WorkerPortError("worker_reconciliation_required", messages=messages)

    backend = WslWorkerBackend(config, launcher=PartialLauncher())
    result = backend.train(
        context=LibraryExecutionContext(
            IDENTITY,
            "run-wsl-partial",
            RuntimeOperation.TRAIN,
            config.target_class,
        ),
        model_source=config.host_model_source,
        tokenizer_source=config.host_tokenizer_source,
        rendered_dataset=b'{"text":"Synthetic WSL input"}\n',
        resolution=_resolution(),
        resume_checkpoint=None,
        on_progress=lambda step, loss: observed_progress.append((step, loss)),
        on_checkpoint=observed_checkpoints.append,
        on_heartbeat=observed_heartbeats.append,
        cancellation_requested=lambda: False,
        interruption_requested=lambda: False,
    )

    assert result.interrupted is True
    assert result.disconnected is True
    assert observed_progress == [(1, 23)]
    assert [item.payload for item in observed_checkpoints] == [checkpoint]
    assert observed_heartbeats == [1]
    assert len(result.transport_receipts) == 2


def test_worker_response_cannot_reference_output_outside_operation_prefix(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class EscapingLauncher(WslWorkerLauncher):
        def launch(self, spec, invocation, **kwargs):
            del spec, kwargs
            location = PortableLocation("other-operation/outputs/capability.json")
            manifest = build_transfer_manifest(
                TransferDirection.WORKER_TO_HOST,
                {location: ("capability_profile", b"{}")},
            )
            stage_transfer(tmp_path, manifest, {location: b"{}"})
            response = WorkerResponse(
                invocation.context,
                "completed",
                manifest,
                {"capability_location": location.to_dict()},
            )
            return WorkerLaunchResult(response, (), False)

    backend = WslWorkerBackend(config, launcher=EscapingLauncher())
    with pytest.raises(LibraryRuntimeError, match="wsl_response_location_invalid"):
        backend.probe()
