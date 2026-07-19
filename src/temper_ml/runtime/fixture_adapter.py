"""Deterministic, offline fixture adapter training over frozen Temper records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import Mapping

from temper_ml.domain.artifacts import BundleManifest, build_bytes_bundle_manifest
from temper_ml.domain.datasets import DatasetVersion
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordValidationError,
    identity_fields,
    record_reference,
)
from temper_ml.domain.runs import ResolvedRuntimeRequest, Run
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)
from temper_ml.runtime.protocol import RuntimeMessage
from temper_ml.runtime.staging import TransferReceipt

FIXTURE_RUNTIME_PROJECTION = HashProjection("runtime.fixture_adapter", "v1")
FIXTURE_TRAINING_STATE_PROJECTION = HashProjection(
    "runtime.fixture_training_state", "v1"
)
FIXTURE_RUNTIME_IDENTITY = content_identity(
    FIXTURE_RUNTIME_PROJECTION,
    {
        "schema_version": "v1",
        "runtime": "temper_fixture_adapter",
        "network": False,
        "accelerator": False,
        "external_trainer": False,
        "external_dashboard": False,
    },
)
FIXTURE_ARTIFACT_MEMBERS = (
    "adapter.bin",
    "adapter_config.json",
    "provenance.json",
)


class FixtureAdapterError(RuntimeError):
    """Public-safe fixture runtime failure represented by one stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class FixtureTermination(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class FixtureControl:
    """Deterministic test control for a cancellation or interruption boundary."""

    cancel_after_step: int | None = None
    interrupt_after_step: int | None = None

    def __post_init__(self) -> None:
        for value in (self.cancel_after_step, self.interrupt_after_step):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise FixtureAdapterError("fixture_control_invalid")
        if self.cancel_after_step is not None and self.interrupt_after_step is not None:
            raise FixtureAdapterError("fixture_control_conflict")


@dataclass(frozen=True)
class FixtureAdapterRequest:
    """Actual immutable manifests and local dataset bytes consumed by the fixture."""

    experiment: Experiment
    recipe_resolution: RecipeResolution
    dataset_version: DatasetVersion
    rendered_dataset: bytes
    runtime_request: ResolvedRuntimeRequest
    run: Run

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, Experiment):
            raise FixtureAdapterError("fixture_experiment_invalid")
        if not isinstance(self.recipe_resolution, RecipeResolution):
            raise FixtureAdapterError("fixture_resolution_invalid")
        if not isinstance(self.dataset_version, DatasetVersion):
            raise FixtureAdapterError("fixture_dataset_invalid")
        if not isinstance(self.rendered_dataset, bytes):
            raise FixtureAdapterError("fixture_dataset_bytes_invalid")
        if not isinstance(self.runtime_request, ResolvedRuntimeRequest):
            raise FixtureAdapterError("fixture_request_invalid")
        if not isinstance(self.run, Run):
            raise FixtureAdapterError("fixture_run_invalid")
        request = self.runtime_request
        if request.experiment != record_reference(self.experiment):
            raise FixtureAdapterError("fixture_request_experiment_mismatch")
        if request.experiment_manifest_identity != self.experiment.manifest_identity:
            raise FixtureAdapterError("fixture_manifest_identity_mismatch")
        if request.recipe_resolution != record_reference(self.recipe_resolution):
            raise FixtureAdapterError("fixture_request_resolution_mismatch")
        if self.experiment.recipe_resolution != record_reference(
            self.recipe_resolution
        ):
            raise FixtureAdapterError("fixture_experiment_resolution_mismatch")
        if self.experiment.dataset_version != self.dataset_version.identity:
            raise FixtureAdapterError("fixture_experiment_dataset_mismatch")
        if request.dataset_version_identity != self.dataset_version.identity:
            raise FixtureAdapterError("fixture_request_dataset_mismatch")
        if (
            request.rendered_dataset_identity
            != self.dataset_version.rendered_bytes_identity
        ):
            raise FixtureAdapterError("fixture_rendered_dataset_mismatch")
        if request.rendered_dataset_byte_count != len(self.rendered_dataset):
            raise FixtureAdapterError("fixture_rendered_dataset_size_mismatch")
        if (
            hashlib.sha256(self.rendered_dataset).hexdigest()
            != request.rendered_dataset_identity.value
        ):
            raise FixtureAdapterError("fixture_rendered_dataset_identity_mismatch")
        if self.run.experiment != record_reference(self.experiment):
            raise FixtureAdapterError("fixture_run_experiment_mismatch")
        if self.run.request_identity != request.identity:
            raise FixtureAdapterError("fixture_run_request_mismatch")
        if self.run.training_state_identity != request.training_state_identity:
            raise FixtureAdapterError("fixture_run_training_state_mismatch")
        if self.run.runtime_identity != request.runtime_identity:
            raise FixtureAdapterError("fixture_run_runtime_mismatch")


@dataclass(frozen=True)
class FixtureProgress:
    step: int
    total_steps: int
    loss_microunits: int

    def to_dict(self) -> dict[str, int]:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "loss_microunits": self.loss_microunits,
        }


@dataclass(frozen=True)
class FixtureLog:
    ordinal: int
    code: str
    step: int

    def to_dict(self) -> dict[str, object]:
        return {"ordinal": self.ordinal, "code": self.code, "step": self.step}


@dataclass(frozen=True)
class FixtureCheckpoint:
    step: int
    training_state_identity: ContentIdentity
    checkpoint_identity: ContentIdentity
    payload: bytes
    resume_compatible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.resume_compatible, bool):
            raise FixtureAdapterError("fixture_checkpoint_invalid")

    def to_receipt(self) -> dict[str, object]:
        return {
            "step": self.step,
            "training_state_identity": identity_fields(self.training_state_identity),
            "checkpoint_identity": identity_fields(self.checkpoint_identity),
            "byte_count": len(self.payload),
            "resume_compatible": self.resume_compatible,
        }


@dataclass(frozen=True)
class FixtureAdapterOutput:
    termination: FixtureTermination
    progress: tuple[FixtureProgress, ...]
    checkpoints: tuple[FixtureCheckpoint, ...]
    logs: tuple[FixtureLog, ...]
    members: Mapping[str, bytes]
    bundle_manifest: BundleManifest | None
    runtime_messages: tuple[RuntimeMessage, ...] = ()
    transfer_receipts: tuple[TransferReceipt, ...] = ()

    def __post_init__(self) -> None:
        copied = {path: bytes(value) for path, value in self.members.items()}
        object.__setattr__(self, "members", MappingProxyType(copied))
        if self.termination is FixtureTermination.COMPLETED:
            if tuple(sorted(copied)) != FIXTURE_ARTIFACT_MEMBERS:
                raise FixtureAdapterError("fixture_artifact_structure_invalid")
            expected = build_bytes_bundle_manifest(copied)
            if self.bundle_manifest != expected:
                raise FixtureAdapterError("fixture_bundle_manifest_mismatch")
        elif copied or self.bundle_manifest is not None:
            raise FixtureAdapterError("fixture_terminal_artifact_invalid")
        if not isinstance(self.runtime_messages, tuple) or any(
            not isinstance(message, RuntimeMessage) for message in self.runtime_messages
        ):
            raise FixtureAdapterError("fixture_runtime_messages_invalid")
        if not isinstance(self.transfer_receipts, tuple) or any(
            not isinstance(receipt, TransferReceipt)
            for receipt in self.transfer_receipts
        ):
            raise FixtureAdapterError("fixture_transfer_receipts_invalid")

    @property
    def completed(self) -> bool:
        return self.termination is FixtureTermination.COMPLETED


class FixtureAdapter:
    """Pure deterministic trainer used to exercise the production runtime port."""

    runtime_identity = FIXTURE_RUNTIME_IDENTITY
    runtime_kind = "fixture"

    def training_state_identity(
        self,
        experiment: Experiment,
        resolution: RecipeResolution,
        dataset_version: DatasetVersion,
        step: int,
    ) -> ContentIdentity:
        return fixture_training_state_identity(
            experiment, resolution, dataset_version, step
        )

    def validate_checkpoint(
        self,
        request: FixtureAdapterRequest,
        checkpoint: FixtureCheckpoint,
    ) -> None:
        if not isinstance(request, FixtureAdapterRequest) or not isinstance(
            checkpoint, FixtureCheckpoint
        ):
            raise FixtureAdapterError("fixture_checkpoint_invalid")
        expected_state = fixture_training_state_identity(
            request.experiment,
            request.recipe_resolution,
            request.dataset_version,
            checkpoint.step,
        )
        expected_producing_run = (
            request.runtime_request.resume_from_run
            if request.runtime_request.resume_from_run is not None
            else record_reference(request.run)
        )
        try:
            payload = loads_canonical_json(checkpoint.payload)
        except (CanonicalJsonError, TypeError, ValueError):
            raise FixtureAdapterError("fixture_checkpoint_invalid") from None
        expected_payload = {
            "schema_version": "v1",
            "producing_run": expected_producing_run.to_dict(),
            "experiment_manifest_identity": identity_fields(
                request.experiment.manifest_identity
            ),
            "recipe_resolution": record_reference(request.recipe_resolution).to_dict(),
            "dataset_version_identity": identity_fields(
                request.dataset_version.identity
            ),
            "rendered_dataset_identity": identity_fields(
                request.dataset_version.rendered_bytes_identity
            ),
            "execution_target": request.runtime_request.execution_target.to_dict(),
            "runtime_identity": identity_fields(FIXTURE_RUNTIME_IDENTITY),
            "step": checkpoint.step,
            "training_state_identity": identity_fields(expected_state),
        }
        if (
            payload != expected_payload
            or dumps_canonical_json(payload) != checkpoint.payload
            or checkpoint.training_state_identity != expected_state
            or checkpoint.checkpoint_identity
            != ContentIdentity("sha256", hashlib.sha256(checkpoint.payload).hexdigest())
            or checkpoint.resume_compatible
            != (checkpoint.step < request.recipe_resolution.training_steps)
        ):
            raise FixtureAdapterError("fixture_checkpoint_invalid")

    def execute(
        self,
        request: FixtureAdapterRequest,
        *,
        control: FixtureControl | None = None,
        resume_checkpoint: FixtureCheckpoint | None = None,
    ) -> FixtureAdapterOutput:
        if not isinstance(request, FixtureAdapterRequest):
            raise FixtureAdapterError("fixture_request_invalid")
        require_fixture_request(request)
        resuming = request.runtime_request.starting_step > 0
        if resuming != (resume_checkpoint is not None):
            raise FixtureAdapterError("fixture_resume_checkpoint_missing")
        if resume_checkpoint is not None:
            self.validate_checkpoint(request, resume_checkpoint)
            if (
                request.runtime_request.resume_checkpoint_identity
                != resume_checkpoint.checkpoint_identity
            ):
                raise FixtureAdapterError("fixture_resume_checkpoint_mismatch")
        active_control = control if control is not None else FixtureControl()
        if not isinstance(active_control, FixtureControl):
            raise FixtureAdapterError("fixture_control_invalid")
        start = request.runtime_request.starting_step
        total = request.recipe_resolution.training_steps
        if active_control.cancel_after_step is not None and not (
            start < active_control.cancel_after_step <= total
        ):
            raise FixtureAdapterError("fixture_control_out_of_range")
        if active_control.interrupt_after_step is not None and not (
            start < active_control.interrupt_after_step < total
        ):
            raise FixtureAdapterError("fixture_control_out_of_range")

        progress: list[FixtureProgress] = []
        checkpoints: list[FixtureCheckpoint] = []
        logs: list[FixtureLog] = [FixtureLog(1, "fixture_runtime_started", start)]
        for step in range(start + 1, total + 1):
            progress.append(
                FixtureProgress(
                    step=step,
                    total_steps=total,
                    loss_microunits=_loss_microunits(request, step),
                )
            )
            logs.append(FixtureLog(len(logs) + 1, "fixture_step_completed", step))
            checkpoint_due = step % request.recipe_resolution.checkpoint_cadence == 0
            interrupted = active_control.interrupt_after_step == step
            if checkpoint_due or interrupted:
                checkpoints.append(build_fixture_checkpoint(request, step))
                logs.append(FixtureLog(len(logs) + 1, "fixture_checkpoint_saved", step))
            if active_control.cancel_after_step == step:
                logs.append(FixtureLog(len(logs) + 1, "fixture_cancelled", step))
                return FixtureAdapterOutput(
                    FixtureTermination.CANCELLED,
                    tuple(progress),
                    tuple(checkpoints),
                    tuple(logs),
                    {},
                    None,
                )
            if interrupted:
                logs.append(FixtureLog(len(logs) + 1, "fixture_interrupted", step))
                return FixtureAdapterOutput(
                    FixtureTermination.INTERRUPTED,
                    tuple(progress),
                    tuple(checkpoints),
                    tuple(logs),
                    {},
                    None,
                )

        members = _artifact_members(request)
        logs.append(FixtureLog(len(logs) + 1, "fixture_artifact_emitted", total))
        return FixtureAdapterOutput(
            FixtureTermination.COMPLETED,
            tuple(progress),
            tuple(checkpoints),
            tuple(logs),
            members,
            build_bytes_bundle_manifest(members),
        )


def fixture_training_state_identity(
    experiment: Experiment,
    resolution: RecipeResolution,
    dataset_version: DatasetVersion,
    step: int,
) -> ContentIdentity:
    """Derive one exact training-state identity from scientific inputs and step."""

    if not isinstance(experiment, Experiment) or not isinstance(
        resolution, RecipeResolution
    ):
        raise FixtureAdapterError("fixture_training_state_input_invalid")
    if not isinstance(dataset_version, DatasetVersion):
        raise FixtureAdapterError("fixture_training_state_input_invalid")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise FixtureAdapterError("fixture_training_state_step_invalid")
    if step > resolution.training_steps:
        raise FixtureAdapterError("fixture_training_state_step_invalid")
    return content_identity(
        FIXTURE_TRAINING_STATE_PROJECTION,
        {
            "schema_version": "v1",
            "experiment": experiment.to_dict(),
            "recipe_resolution": resolution.to_dict(),
            "dataset_version_identity": identity_fields(dataset_version.identity),
            "rendered_dataset_identity": identity_fields(
                dataset_version.rendered_bytes_identity
            ),
            "step": step,
        },
    )


def fixture_adapter_bytes(
    experiment: Experiment,
    resolution: RecipeResolution,
    dataset_version: DatasetVersion,
) -> bytes:
    """Recompute the exact final adapter payload for integrity verification."""

    final_state = fixture_training_state_identity(
        experiment,
        resolution,
        dataset_version,
        resolution.training_steps,
    )
    adapter_preimage = dumps_canonical_json(
        {
            "schema_version": "v1",
            "experiment": experiment.to_dict(),
            "recipe_resolution": resolution.to_dict(),
            "rendered_dataset_identity": identity_fields(
                dataset_version.rendered_bytes_identity
            ),
            "final_training_state_identity": identity_fields(final_state),
        }
    )
    return (
        b"TEMPER-FIXTURE-ADAPTER-v1\n"
        + hashlib.sha256(adapter_preimage).digest()
        + b"\n"
    )


def build_fixture_checkpoint(
    request: FixtureAdapterRequest, step: int
) -> FixtureCheckpoint:
    """Build the exact portable checkpoint bytes emitted for one completed step."""

    state = fixture_training_state_identity(
        request.experiment,
        request.recipe_resolution,
        request.dataset_version,
        step,
    )
    payload = dumps_canonical_json(
        {
            "schema_version": "v1",
            "producing_run": record_reference(request.run).to_dict(),
            "experiment_manifest_identity": identity_fields(
                request.experiment.manifest_identity
            ),
            "recipe_resolution": record_reference(request.recipe_resolution).to_dict(),
            "dataset_version_identity": identity_fields(
                request.dataset_version.identity
            ),
            "rendered_dataset_identity": identity_fields(
                request.dataset_version.rendered_bytes_identity
            ),
            "execution_target": request.runtime_request.execution_target.to_dict(),
            "runtime_identity": identity_fields(FIXTURE_RUNTIME_IDENTITY),
            "step": step,
            "training_state_identity": identity_fields(state),
        }
    )
    return FixtureCheckpoint(
        step=step,
        training_state_identity=state,
        checkpoint_identity=ContentIdentity(
            "sha256", hashlib.sha256(payload).hexdigest()
        ),
        payload=payload,
        resume_compatible=step < request.recipe_resolution.training_steps,
    )


def _loss_microunits(request: FixtureAdapterRequest, step: int) -> int:
    preimage = dumps_canonical_json(
        {
            "experiment_manifest_identity": identity_fields(
                request.experiment.manifest_identity
            ),
            "recipe_resolution_identity": identity_fields(
                request.recipe_resolution.identity
            ),
            "rendered_dataset_identity": identity_fields(
                request.dataset_version.rendered_bytes_identity
            ),
            "step": step,
        }
    )
    jitter = int.from_bytes(hashlib.sha256(preimage).digest()[:2], "big") % 1000
    return ((request.recipe_resolution.training_steps - step + 1) * 1_000_000) + jitter


def _artifact_members(request: FixtureAdapterRequest) -> dict[str, bytes]:
    final_state = fixture_training_state_identity(
        request.experiment,
        request.recipe_resolution,
        request.dataset_version,
        request.recipe_resolution.training_steps,
    )
    adapter_bytes = fixture_adapter_bytes(
        request.experiment,
        request.recipe_resolution,
        request.dataset_version,
    )
    adapter_identity = ContentIdentity(
        "sha256", hashlib.sha256(adapter_bytes).hexdigest()
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
            "compatibility_group": (request.experiment.compatibility_group.to_dict()),
            "adapter_identity": identity_fields(adapter_identity),
            "runtime_identity": identity_fields(FIXTURE_RUNTIME_IDENTITY),
            "training_steps": request.recipe_resolution.training_steps,
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
            "recipe_resolution": record_reference(request.recipe_resolution).to_dict(),
            "dataset_version_identity": identity_fields(
                request.dataset_version.identity
            ),
            "final_training_state_identity": identity_fields(final_state),
        }
    )
    return {
        "adapter.bin": adapter_bytes,
        "adapter_config.json": config,
        "provenance.json": provenance,
    }


def require_fixture_request(request: FixtureAdapterRequest) -> None:
    """Re-run construction invariants at a runtime boundary."""

    try:
        request.__post_init__()
    except (FixtureAdapterError, RecordValidationError, TypeError, ValueError):
        raise FixtureAdapterError("fixture_request_invalid") from None
    if request.runtime_request.runtime_identity != FIXTURE_RUNTIME_IDENTITY:
        raise FixtureAdapterError("fixture_runtime_identity_mismatch")
    expected_state = fixture_training_state_identity(
        request.experiment,
        request.recipe_resolution,
        request.dataset_version,
        request.runtime_request.starting_step,
    )
    if request.runtime_request.training_state_identity != expected_state:
        raise FixtureAdapterError("fixture_training_state_mismatch")
