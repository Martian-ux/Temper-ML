"""Deterministic local double for the library runtime contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import identity_fields, record_reference
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
from temper_ml.runtime.protocol import RuntimeOperation
from temper_ml.store.canonical_json import (
    dumps_canonical_json,
    loads_canonical_json,
)


class DeterministicLibraryBackend:
    """Exercise every runtime callback without ML libraries, a GPU, or network."""

    def __init__(self, capability: LibraryCapability) -> None:
        if not isinstance(capability, LibraryCapability):
            raise LibraryRuntimeError("library_capability_invalid")
        self._capability = capability
        self.train_calls = 0
        self.inference_calls = 0
        self.inference_operations: list[RuntimeOperation] = []

    def probe(self) -> LibraryCapability:
        return self._capability

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
        del model_source, tokenizer_source
        if (
            not isinstance(context, LibraryExecutionContext)
            or context.operation is not RuntimeOperation.TRAIN
            or not isinstance(rendered_dataset, bytes)
            or not rendered_dataset
            or not isinstance(resolution, RecipeResolution)
        ):
            raise LibraryRuntimeError("library_double_request_invalid")
        self.train_calls += 1
        start = _checkpoint_step(resume_checkpoint, resolution)
        progress: list[tuple[int, int]] = []
        checkpoints: list[LibraryCheckpointPayload] = []
        dataset_digest = hashlib.sha256(rendered_dataset).hexdigest()
        for step in range(start + 1, resolution.training_steps + 1):
            if interruption_requested():
                return LibraryTrainingResult(
                    None,
                    None,
                    None,
                    tuple(progress),
                    tuple(checkpoints),
                    interrupted=True,
                )
            if cancellation_requested():
                return LibraryTrainingResult(
                    None,
                    None,
                    None,
                    tuple(progress),
                    tuple(checkpoints),
                    cancelled=True,
                )
            loss = int(dataset_digest[:8], 16) % 1_000_000 + step
            progress.append((step, loss))
            on_progress(step, loss)
            on_heartbeat(step)
            if (
                step % resolution.checkpoint_cadence == 0
                or step == resolution.training_steps
            ):
                checkpoint = LibraryCheckpointPayload(
                    step, _checkpoint_bytes(step, resolution)
                )
                checkpoints.append(checkpoint)
                on_checkpoint(checkpoint)
            if interruption_requested():
                return LibraryTrainingResult(
                    None,
                    None,
                    None,
                    tuple(progress),
                    tuple(checkpoints),
                    interrupted=True,
                )
        payload = dumps_canonical_json(
            {
                "schema_version": "v1",
                "kind": "deterministic_library_adapter",
                "request_identity": identity_fields(context.request_identity),
                "recipe_resolution": record_reference(resolution).to_dict(),
                "rendered_dataset_sha256": dataset_digest,
            }
        )
        return LibraryTrainingResult(
            payload,
            "safetensors",
            {
                "peft_type": "lora",
                "task_type": "causal_lm",
                "bias": "none",
                "rank": resolution.rank,
                "alpha": resolution.alpha,
                "dropout": str(resolution.dropout),
                "payload_format": "safetensors",
            },
            tuple(progress),
            tuple(checkpoints),
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
        del model_source, tokenizer_source
        if (
            not isinstance(context, LibraryExecutionContext)
            or context.operation
            not in {
                RuntimeOperation.EVALUATE,
                RuntimeOperation.INFER_FOCUSED,
                RuntimeOperation.INFER_BATCH,
            }
            or not adapter_payload
            or adapter_payload_format != "safetensors"
            or not isinstance(resolution, RecipeResolution)
            or not isinstance(settings, InferenceSettings)
            or not inputs
        ):
            raise LibraryRuntimeError("library_double_inference_invalid")
        self.inference_calls += 1
        self.inference_operations.append(context.operation)
        adapter_digest = hashlib.sha256(adapter_payload).hexdigest()
        outputs = tuple(
            "library-double-"
            + hashlib.sha256(
                dumps_canonical_json(
                    {
                        "schema_version": "v1",
                        "operation": context.operation.value,
                        "adapter_sha256": adapter_digest,
                        "input": value,
                        "settings": settings.to_dict(),
                    }
                )
            ).hexdigest()[:24]
            for value in inputs
        )
        return LibraryInferenceResult(outputs, self._capability.library_versions)


def _checkpoint_bytes(step: int, resolution: RecipeResolution) -> bytes:
    return dumps_canonical_json(
        {
            "schema_version": "v1",
            "step": step,
            "recipe_resolution": record_reference(resolution).to_dict(),
        }
    )


def _checkpoint_step(payload: bytes | None, resolution: RecipeResolution) -> int:
    if payload is None:
        return 0
    try:
        value = loads_canonical_json(payload)
    except (TypeError, ValueError):
        raise LibraryRuntimeError("library_checkpoint_restore_failed") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "step", "recipe_resolution"}
        or value["schema_version"] != "v1"
        or value["recipe_resolution"] != record_reference(resolution).to_dict()
        or isinstance(value["step"], bool)
        or not isinstance(value["step"], int)
        or value["step"] < 1
        or value["step"] >= resolution.training_steps
        or dumps_canonical_json(value) != payload
    ):
        raise LibraryRuntimeError("library_checkpoint_restore_failed")
    return value["step"]
