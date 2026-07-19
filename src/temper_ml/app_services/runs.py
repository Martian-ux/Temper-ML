"""Temper-owned run lifecycle, recovery, and artifact-ingestion services."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Mapping, NoReturn

from temper_ml.app_services._records import (
    require_no_conflicting_logical_revision,
    write_record_idempotently,
)
from temper_ml.app_services.datasets import PreparedDataset
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactAvailability,
    ArtifactContentKind,
    AvailabilityState,
    StorageReference,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import (
    CompatibilityGroup,
    ResumeCheckpoint,
    ResumeRequest,
    check_resume_compatibility,
)
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordValidationError,
    TypedRecord,
    identity_fields,
    parse_identity,
    record_reference,
    require_identifier,
)
from temper_ml.domain.runs import (
    EvaluationMode,
    ResolvedRuntimeRequest,
    Run,
)
from temper_ml.filesystem import UnsafeFilesystemPath, ensure_safe_directory
from temper_ml.runtime.artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactIntegrityExpectation,
    ArtifactIntegrityResult,
    verify_artifact_bundle,
)
from temper_ml.runtime.fixture_adapter import (
    FixtureAdapter,
    FixtureAdapterError,
    FixtureAdapterOutput,
    FixtureAdapterRequest,
    FixtureCheckpoint,
    FixtureControl,
    FixtureTermination,
)
from temper_ml.runtime.library_adapter import LibraryAdapterExecutionError
from temper_ml.runtime.library_backend import LibraryRuntimeError
from temper_ml.runtime.ownership import (
    RunOwnershipError,
    RunOwnershipLease,
    claim_run_ownership,
)
from temper_ml.runtime.preflight import (
    PreflightError,
    PreflightEstimate,
    PreflightResult,
    preflight,
)
from temper_ml.runtime.controller import (
    ControllerSnapshot,
    RuntimeControllerError,
    SerializedRunController,
)
from temper_ml.runtime.protocol import RuntimeMessage, RuntimeProtocolError
from temper_ml.runtime.staging import StagingError, TransferDirection, TransferReceipt
from temper_ml.store.evidence import (
    EvidenceError,
    EvidenceExists,
    TypedEvidenceStore,
)
from temper_ml.store.event_stream import EventRequest, StoredEvent
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes, write_once_bytes

PREFLIGHT_EVIDENCE_PROJECTION = HashProjection("runtime.preflight_evidence", "v1")
ARTIFACT_LINEAGE_PROJECTION = HashProjection("artifact.runtime_lineage", "v1")
RUN_OWNERSHIP_PROJECTION = HashProjection("runtime.run_ownership_claim", "v1")
RUNTIME_OUTPUT_DIRECTORY = ".temper-fixture-output"


class RunLifecycleStatus(str, Enum):
    PREFLIGHT_BLOCKED = "preflight_blocked"
    RUNNING = "running"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            RunLifecycleStatus.PREFLIGHT_BLOCKED,
            RunLifecycleStatus.CANCELLED,
            RunLifecycleStatus.INTERRUPTED,
            RunLifecycleStatus.COMPLETED,
            RunLifecycleStatus.FAILED,
        }


@dataclass(frozen=True)
class RunLaunchRequest:
    """Complete local inputs for one new immutable execution attempt."""

    run_id: str
    request_id: str
    artifact_id: str
    experiment: Experiment
    recipe_resolution: RecipeResolution
    prepared_dataset: PreparedDataset
    base_model_revision: BaseModelRevision
    compatibility_group: CompatibilityGroup
    hardware_requirements: HardwareRequirements
    execution_target: ExecutionTarget
    hardware_capability_profile: HardwareCapabilityProfile
    estimate: PreflightEstimate
    evaluation_mode: EvaluationMode = EvaluationMode.NO_QUALITY_EVALUATION

    def __post_init__(self) -> None:
        for field in ("run_id", "request_id", "artifact_id"):
            try:
                require_identifier(field, getattr(self, field))
            except RecordValidationError:
                raise ApplicationServiceError("run_launch_request_invalid") from None
        expected = (
            (self.experiment, Experiment),
            (self.recipe_resolution, RecipeResolution),
            (self.prepared_dataset, PreparedDataset),
            (self.base_model_revision, BaseModelRevision),
            (self.compatibility_group, CompatibilityGroup),
            (self.hardware_requirements, HardwareRequirements),
            (self.execution_target, ExecutionTarget),
            (self.hardware_capability_profile, HardwareCapabilityProfile),
            (self.estimate, PreflightEstimate),
        )
        if any(not isinstance(value, kind) for value, kind in expected):
            raise ApplicationServiceError("run_launch_request_invalid")
        if not isinstance(self.evaluation_mode, EvaluationMode):
            raise ApplicationServiceError("run_evaluation_mode_invalid")


@dataclass(frozen=True)
class RunRecoveryRequest:
    """A new run request tied to one retained checkpoint from an interrupted run."""

    launch: RunLaunchRequest
    interrupted_run: Run
    checkpoint_identity: ContentIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.launch, RunLaunchRequest) or not isinstance(
            self.interrupted_run, Run
        ):
            raise ApplicationServiceError("run_recovery_request_invalid")
        if not isinstance(self.checkpoint_identity, ContentIdentity):
            raise ApplicationServiceError("run_recovery_request_invalid")


@dataclass(frozen=True)
class RunExecutionResult:
    """One coherent terminal attempt and any verified canonical artifact."""

    run: Run
    runtime_request: ResolvedRuntimeRequest
    preflight: PreflightResult
    status: RunLifecycleStatus
    checkpoints: tuple[FixtureCheckpoint, ...]
    artifact: Artifact | None = None
    availability: ArtifactAvailability | None = None
    integrity: ArtifactIntegrityResult | None = None

    def __post_init__(self) -> None:
        if (
            not self.status.terminal
            or self.status is RunLifecycleStatus.PREFLIGHT_BLOCKED
        ):
            raise ApplicationServiceError("run_result_not_terminal")
        verified = (
            self.artifact is not None
            and self.availability is not None
            and self.integrity is not None
        )
        if self.status is RunLifecycleStatus.COMPLETED and not verified:
            raise ApplicationServiceError("run_result_artifact_missing")
        if self.status is not RunLifecycleStatus.COMPLETED and (
            self.artifact is not None
            or self.availability is not None
            or self.integrity is not None
        ):
            raise ApplicationServiceError("run_result_artifact_forbidden")

    @property
    def verified_artifact(self) -> bool:
        return self.status is RunLifecycleStatus.COMPLETED and self.artifact is not None

    def to_view(self) -> dict[str, object]:
        value: dict[str, object] = {
            "status": self.status.value,
            "run": record_reference(self.run).to_dict(),
            "runtime_request": record_reference(self.runtime_request).to_dict(),
            "preflight": self.preflight.to_view(),
            "checkpoint_count": len(self.checkpoints),
            "verified_artifact": self.verified_artifact,
        }
        if self.artifact is not None and self.integrity is not None:
            value["artifact"] = record_reference(self.artifact).to_dict()
            value["artifact_integrity"] = self.integrity.to_receipt()
        return value


class RunService:
    """Own the complete run lifecycle and all canonical evidence writes."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        adapter: FixtureAdapter | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.store = TypedEvidenceStore(self.project_root)
        self.adapter = adapter if adapter is not None else FixtureAdapter()
        if not isinstance(self.adapter, FixtureAdapter):
            raise ApplicationServiceError("fixture_adapter_invalid")

    def launch(
        self,
        request: RunLaunchRequest,
        *,
        control: FixtureControl | None = None,
    ) -> RunExecutionResult:
        """Preflight, freeze, launch, ingest, and terminate one first attempt."""

        return self._launch(
            request,
            control=control,
            attempt_number=1,
            retry_of=None,
            recovery_checkpoint=None,
        )

    def recover(
        self,
        request: RunRecoveryRequest,
        *,
        control: FixtureControl | None = None,
    ) -> RunExecutionResult:
        """Create a new attempt only from an exact retained compatible checkpoint."""

        if not isinstance(request, RunRecoveryRequest):
            raise ApplicationServiceError("run_recovery_request_invalid")
        prior = self._require_exact_record(request.interrupted_run)
        if not isinstance(prior, Run):
            raise ApplicationServiceError("run_recovery_source_invalid")
        if self.status(prior.run_id) is not RunLifecycleStatus.INTERRUPTED:
            raise ApplicationServiceError("run_recovery_source_not_interrupted")
        prior_request = self._runtime_request_for_identity(prior.request_identity)
        launch = request.launch
        if (
            prior.experiment != record_reference(launch.experiment)
            or prior.experiment_manifest_identity != launch.experiment.manifest_identity
        ):
            raise ApplicationServiceError("run_recovery_experiment_mismatch")
        event = self._checkpoint_event(prior.run_id, request.checkpoint_identity)
        if event.payload.get("resume_compatible") is not True:
            raise ApplicationServiceError("run_recovery_checkpoint_incompatible")
        if not self._checkpoint_resume_available(prior.run_id, event):
            raise ApplicationServiceError("run_recovery_checkpoint_unavailable")
        step = _event_positive_int(event, "step")
        state_identity = _event_identity(event, "training_state_identity")
        provisional = FixtureCheckpoint(
            step,
            state_identity,
            request.checkpoint_identity,
            b"",
            True,
        )
        path = self._checkpoint_path(prior, provisional)
        try:
            retained = read_stable_bytes(path)
        except SafeIoError:
            raise ApplicationServiceError(
                "run_recovery_checkpoint_unavailable"
            ) from None
        checkpoint = FixtureCheckpoint(
            step,
            state_identity,
            request.checkpoint_identity,
            retained,
            True,
        )
        if (
            _event_positive_int(event, "byte_count") != len(retained)
            or ContentIdentity("sha256", hashlib.sha256(retained).hexdigest())
            != request.checkpoint_identity
        ):
            raise ApplicationServiceError("run_recovery_checkpoint_corrupt")
        adapter_request = FixtureAdapterRequest(
            experiment=launch.experiment,
            recipe_resolution=launch.recipe_resolution,
            dataset_version=launch.prepared_dataset.version,
            rendered_dataset=launch.prepared_dataset.rendered_bytes,
            runtime_request=prior_request,
            run=prior,
        )
        try:
            expected_state = self.adapter.training_state_identity(
                launch.experiment,
                launch.recipe_resolution,
                launch.prepared_dataset.version,
                step,
            )
            if state_identity != expected_state:
                raise ApplicationServiceError("run_recovery_checkpoint_mismatch")
            self.adapter.validate_checkpoint(adapter_request, checkpoint)
        except (FixtureAdapterError, LibraryRuntimeError):
            raise ApplicationServiceError("run_recovery_checkpoint_mismatch") from None
        decision = check_resume_compatibility(
            ResumeCheckpoint(
                experiment_manifest_identity=prior.experiment_manifest_identity,
                recipe_resolution=prior_request.recipe_resolution,
                training_state_identity=checkpoint.training_state_identity,
                execution_target=prior.execution_target,
                available=True,
            ),
            ResumeRequest(
                experiment_manifest_identity=launch.experiment.manifest_identity,
                recipe_resolution=record_reference(launch.recipe_resolution),
                training_state_identity=checkpoint.training_state_identity,
                execution_target=record_reference(launch.execution_target),
            ),
        )
        if not decision.compatible:
            raise ApplicationServiceError("run_recovery_incompatible")
        return self._launch(
            launch,
            control=control,
            attempt_number=prior.attempt_number + 1,
            retry_of=prior,
            recovery_checkpoint=checkpoint,
        )

    def reopen_completed(self, request: RunLaunchRequest) -> RunExecutionResult:
        """Reopen one exact completed first attempt without executing it again."""

        if not isinstance(request, RunLaunchRequest):
            raise ApplicationServiceError("run_launch_request_invalid")
        if request.evaluation_mode is not EvaluationMode.NO_QUALITY_EVALUATION:
            raise ApplicationServiceError("run_evaluation_mode_not_supported")
        try:
            self.store.verify()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        self._validate_launch_graph(request)
        try:
            preflight_result = preflight(
                request.recipe_resolution,
                request.hardware_requirements,
                request.execution_target,
                request.hardware_capability_profile,
                request.estimate,
            )
        except PreflightError as exc:
            raise ApplicationServiceError(exc.code) from None
        preflight_identity = content_identity(
            PREFLIGHT_EVIDENCE_PROJECTION, preflight_result.to_view()
        )
        try:
            runtime_request, run = self._build_execution_records(
                request,
                preflight_identity,
                attempt_number=1,
                retry_of=None,
                recovery_checkpoint=None,
            )
        except (
            FixtureAdapterError,
            LibraryRuntimeError,
            RecordValidationError,
            TypeError,
            ValueError,
        ):
            raise ApplicationServiceError("run_existing_conflict") from None
        events = self._events(request.run_id)
        matching_runs = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, Run) and stored.record.run_id == request.run_id
        )
        if not events and not matching_runs:
            raise ApplicationServiceError("run_not_found")
        if not preflight_result.ready or matching_runs != (run,):
            raise ApplicationServiceError("run_existing_conflict")
        try:
            require_no_conflicting_logical_revision(
                self.store,
                runtime_request,
                conflict_code="run_existing_conflict",
            )
            exact_request = self._require_exact_record(runtime_request)
        except ApplicationServiceError:
            raise ApplicationServiceError("run_existing_conflict") from None
        if not isinstance(exact_request, ResolvedRuntimeRequest):
            raise ApplicationServiceError("run_existing_conflict")
        if self.status(run.run_id) is not RunLifecycleStatus.COMPLETED:
            raise ApplicationServiceError("run_existing_not_completed")
        if self.adapter.runtime_kind != "fixture":
            return self._reopen_library_completed(
                request, runtime_request, run, preflight_result
            )

        try:
            adapter_request = FixtureAdapterRequest(
                request.experiment,
                request.recipe_resolution,
                request.prepared_dataset.version,
                request.prepared_dataset.rendered_bytes,
                runtime_request,
                run,
            )
            expected_output = FixtureAdapter().execute(adapter_request)
        except FixtureAdapterError:
            raise ApplicationServiceError("run_existing_lifecycle_conflict") from None
        if not expected_output.completed or expected_output.bundle_manifest is None:
            raise ApplicationServiceError("run_existing_lifecycle_conflict")
        try:
            for checkpoint in expected_output.checkpoints:
                if (
                    read_stable_bytes(self._checkpoint_path(run, checkpoint))
                    != checkpoint.payload
                ):
                    raise ApplicationServiceError("run_existing_checkpoint_conflict")
        except SafeIoError:
            raise ApplicationServiceError("run_existing_checkpoint_conflict") from None

        run_reference = record_reference(run)
        artifacts = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, Artifact)
            and (
                stored.record.artifact_id == request.artifact_id
                or stored.record.producing_run == run_reference
            )
        )
        if (
            len(artifacts) != 1
            or artifacts[0].artifact_id != request.artifact_id
            or artifacts[0].producing_run != run_reference
            or artifacts[0].content_identity != expected_output.bundle_manifest.identity
        ):
            raise ApplicationServiceError("run_existing_artifact_conflict")
        artifact = artifacts[0]
        expectation = ArtifactIntegrityExpectation(
            bundle_identity=expected_output.bundle_manifest.identity,
            producing_run=run,
            runtime_request=runtime_request,
            experiment=request.experiment,
            recipe_resolution=request.recipe_resolution,
            dataset_version=request.prepared_dataset.version,
            base_model_revision=request.base_model_revision,
            compatibility_group=request.compatibility_group,
        )
        try:
            integrity = verify_artifact_bundle(
                self._artifact_root(request.artifact_id), expectation
            )
        except ArtifactIntegrityError as exc:
            raise ApplicationServiceError(exc.code) from None
        try:
            stored_manifest = self.store.read_bundle_manifest(
                integrity.bundle_manifest.identity
            )
        except EvidenceError:
            raise ApplicationServiceError("run_existing_artifact_conflict") from None
        if (
            integrity.bundle_manifest != expected_output.bundle_manifest
            or stored_manifest != integrity.bundle_manifest
        ):
            raise ApplicationServiceError("run_existing_artifact_conflict")
        lineage = content_identity(
            ARTIFACT_LINEAGE_PROJECTION,
            {
                "schema_version": "v1",
                "experiment": record_reference(request.experiment).to_dict(),
                "resolved_runtime_request": record_reference(runtime_request).to_dict(),
                "producing_run": record_reference(run).to_dict(),
            },
        )
        expected_artifact = Artifact(
            artifact_id=request.artifact_id,
            project=request.experiment.project,
            producing_run=record_reference(run),
            adapter_type=request.recipe_resolution.adapter_type,
            content_kind=ArtifactContentKind.BUNDLE,
            content_identity=integrity.bundle_manifest.identity,
            base_model_revision=record_reference(request.base_model_revision),
            tokenizer_identity=request.base_model_revision.tokenizer_identity,
            compatibility_groups=(record_reference(request.compatibility_group),),
            parent_artifacts=(),
            storage_references=(
                StorageReference("project_store", request.artifact_id),
            ),
            integrity_evidence=integrity.evidence_identity,
            provenance=integrity.provenance_identity,
            lineage_evidence=lineage,
        )
        expected_availability = ArtifactAvailability(
            availability_id=f"available-{expected_artifact.identity.value[:24]}",
            artifact=record_reference(expected_artifact),
            state=AvailabilityState.AVAILABLE,
            available_byte_classes=("final_adapter",),
            storage_references=expected_artifact.storage_references,
            checkpoint_resumable=False,
            observed_content_identity=expected_artifact.content_identity,
        )
        availabilities = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, ArtifactAvailability)
            and stored.record.artifact == record_reference(expected_artifact)
        )
        if artifact != expected_artifact or availabilities != (expected_availability,):
            raise ApplicationServiceError("run_existing_artifact_conflict")
        expected_events = [
            EventRequest(
                f"{run.run_id}-preflight",
                "run_preflight_succeeded",
                {
                    "ready": True,
                    "preflight_identity": identity_fields(preflight_identity),
                    "blocking_reasons": [],
                },
            ),
            EventRequest(
                f"{run.run_id}-request-frozen",
                "runtime_request_frozen",
                {
                    "runtime_request_identity": identity_fields(
                        runtime_request.identity
                    ),
                    "experiment_manifest_identity": identity_fields(
                        request.experiment.manifest_identity
                    ),
                    "preflight_identity": identity_fields(preflight_identity),
                    "evaluation_mode": request.evaluation_mode.value,
                    "starting_step": 0,
                },
            ),
            EventRequest(
                f"{run.run_id}-launched",
                "run_launched",
                {
                    "run_identity": identity_fields(run.identity),
                    "runtime_request_identity": identity_fields(
                        runtime_request.identity
                    ),
                    "attempt_number": 1,
                    "fixture_runtime": True,
                },
            ),
        ]
        expected_events.extend(
            EventRequest(
                f"{run.run_id}-progress-{progress.step}",
                "run_progress",
                progress.to_dict(),
            )
            for progress in expected_output.progress
        )
        expected_events.extend(
            EventRequest(
                f"{run.run_id}-checkpoint-{checkpoint.step}",
                "run_checkpoint",
                checkpoint.to_receipt(),
            )
            for checkpoint in expected_output.checkpoints
        )
        expected_events.extend(
            EventRequest(
                f"{run.run_id}-log-{log.ordinal}",
                "run_log",
                log.to_dict(),
            )
            for log in expected_output.logs
        )
        expected_events.extend(
            (
                EventRequest(
                    f"{run.run_id}-artifact-ingestion-started",
                    "artifact_ingestion_started",
                    {
                        "expected_bundle_identity": identity_fields(
                            expected_output.bundle_manifest.identity
                        ),
                        "expected_member_count": len(
                            expected_output.bundle_manifest.members
                        ),
                    },
                ),
                EventRequest(
                    f"{run.run_id}-artifact-ingestion-verified",
                    "artifact_ingestion_verified",
                    {
                        "artifact_identity": identity_fields(artifact.identity),
                        "bundle_identity": identity_fields(artifact.content_identity),
                        "integrity_evidence": identity_fields(
                            integrity.evidence_identity
                        ),
                        "quality_evaluation_required": False,
                    },
                ),
                EventRequest(
                    f"{run.run_id}-completed",
                    "run_completed",
                    {
                        "terminal": True,
                        "verified_artifact": True,
                        "artifact_identity": identity_fields(artifact.identity),
                        "integrity_evidence": identity_fields(
                            integrity.evidence_identity
                        ),
                    },
                ),
            )
        )
        if tuple(event.request_fields() for event in events) != tuple(
            expected.canonical_fields() for expected in expected_events
        ):
            raise ApplicationServiceError("run_existing_lifecycle_conflict")
        return RunExecutionResult(
            run,
            runtime_request,
            preflight_result,
            RunLifecycleStatus.COMPLETED,
            expected_output.checkpoints,
            artifact,
            expected_availability,
            integrity,
        )

    def _reopen_library_completed(
        self,
        request: RunLaunchRequest,
        runtime_request: ResolvedRuntimeRequest,
        run: Run,
        preflight_result: PreflightResult,
    ) -> RunExecutionResult:
        """Reconcile a completed library run without launching the worker again."""

        if (
            run.runtime_identity != self.adapter.runtime_identity
            or runtime_request.runtime_identity != self.adapter.runtime_identity
        ):
            raise ApplicationServiceError("run_existing_runtime_mismatch")
        events = self._events(run.run_id)
        launched = tuple(
            event for event in events if event.event_type == "run_launched"
        )
        completed = tuple(
            event for event in events if event.event_type == "run_completed"
        )
        if len(launched) != 1 or len(completed) != 1 or events[-1] != completed[0]:
            raise ApplicationServiceError("run_existing_lifecycle_conflict")
        if (
            launched[0].payload.get("fixture_runtime") is not False
            or _event_identity(launched[0], "runtime_identity")
            != self.adapter.runtime_identity
        ):
            raise ApplicationServiceError("run_existing_lifecycle_conflict")
        snapshot = self.reconcile_runtime_controller(run.run_id)
        if snapshot.state.value != "completed":
            raise ApplicationServiceError("run_existing_lifecycle_conflict")
        transfer_events = tuple(
            event for event in events if event.event_type == "runtime_transfer_verified"
        )
        try:
            receipt_values: list[TransferReceipt] = []
            for event in transfer_events:
                raw_receipt = event.request_fields().get("payload")
                if not isinstance(raw_receipt, Mapping):
                    raise StagingError("transfer_receipt_invalid")
                receipt_values.append(TransferReceipt.from_dict(raw_receipt))
            receipts = tuple(receipt_values)
        except StagingError:
            raise ApplicationServiceError("run_existing_transfer_conflict") from None
        directions = {receipt.direction for receipt in receipts}
        if directions != {
            TransferDirection.HOST_TO_WORKER,
            TransferDirection.WORKER_TO_HOST,
        }:
            raise ApplicationServiceError("run_existing_transfer_conflict")
        run_reference = record_reference(run)
        artifacts = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, Artifact)
            and (
                stored.record.artifact_id == request.artifact_id
                or stored.record.producing_run == run_reference
            )
        )
        if (
            len(artifacts) != 1
            or artifacts[0].artifact_id != request.artifact_id
            or artifacts[0].producing_run != run_reference
        ):
            raise ApplicationServiceError("run_existing_artifact_conflict")
        artifact = artifacts[0]
        if snapshot.artifact_identity != artifact.content_identity:
            raise ApplicationServiceError("run_existing_artifact_conflict")
        expectation = ArtifactIntegrityExpectation(
            bundle_identity=artifact.content_identity,
            producing_run=run,
            runtime_request=runtime_request,
            experiment=request.experiment,
            recipe_resolution=request.recipe_resolution,
            dataset_version=request.prepared_dataset.version,
            base_model_revision=request.base_model_revision,
            compatibility_group=request.compatibility_group,
        )
        try:
            integrity = verify_artifact_bundle(
                self._artifact_root(request.artifact_id), expectation
            )
            stored_manifest = self.store.read_bundle_manifest(
                integrity.bundle_manifest.identity
            )
        except (ArtifactIntegrityError, EvidenceError) as exc:
            code = getattr(exc, "code", "run_existing_artifact_conflict")
            raise ApplicationServiceError(code) from None
        if stored_manifest != integrity.bundle_manifest:
            raise ApplicationServiceError("run_existing_artifact_conflict")
        lineage = content_identity(
            ARTIFACT_LINEAGE_PROJECTION,
            {
                "schema_version": "v1",
                "experiment": record_reference(request.experiment).to_dict(),
                "resolved_runtime_request": record_reference(runtime_request).to_dict(),
                "producing_run": record_reference(run).to_dict(),
            },
        )
        expected_artifact = Artifact(
            artifact_id=request.artifact_id,
            project=request.experiment.project,
            producing_run=run_reference,
            adapter_type=request.recipe_resolution.adapter_type,
            content_kind=ArtifactContentKind.BUNDLE,
            content_identity=integrity.bundle_manifest.identity,
            base_model_revision=record_reference(request.base_model_revision),
            tokenizer_identity=request.base_model_revision.tokenizer_identity,
            compatibility_groups=(record_reference(request.compatibility_group),),
            parent_artifacts=(),
            storage_references=(
                StorageReference("project_store", request.artifact_id),
            ),
            integrity_evidence=integrity.evidence_identity,
            provenance=integrity.provenance_identity,
            lineage_evidence=lineage,
        )
        expected_availability = ArtifactAvailability(
            availability_id=f"available-{expected_artifact.identity.value[:24]}",
            artifact=record_reference(expected_artifact),
            state=AvailabilityState.AVAILABLE,
            available_byte_classes=("final_adapter",),
            storage_references=expected_artifact.storage_references,
            checkpoint_resumable=False,
            observed_content_identity=expected_artifact.content_identity,
        )
        availabilities = tuple(
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, ArtifactAvailability)
            and stored.record.artifact == record_reference(expected_artifact)
        )
        if artifact != expected_artifact or availabilities != (expected_availability,):
            raise ApplicationServiceError("run_existing_artifact_conflict")
        checkpoints = tuple(
            self._checkpoint_from_event(run, event)
            for event in events
            if event.event_type == "run_checkpoint"
        )
        completed_artifact = completed[0].payload.get("artifact_identity")
        if not isinstance(completed_artifact, Mapping):
            raise ApplicationServiceError("run_existing_lifecycle_conflict")
        if _event_identity(completed[0], "artifact_identity") != artifact.identity:
            raise ApplicationServiceError("run_existing_lifecycle_conflict")
        return RunExecutionResult(
            run,
            runtime_request,
            preflight_result,
            RunLifecycleStatus.COMPLETED,
            checkpoints,
            artifact,
            expected_availability,
            integrity,
        )

    def status(self, run_id: str) -> RunLifecycleStatus:
        """Derive lifecycle status solely from the verified append-only stream."""

        try:
            require_identifier("run_id", run_id)
        except RecordValidationError:
            raise ApplicationServiceError("run_id_invalid") from None
        events = self._events(run_id)
        if not events:
            raise ApplicationServiceError("run_not_found")
        terminal_events = {
            "run_preflight_blocked": RunLifecycleStatus.PREFLIGHT_BLOCKED,
            "run_cancelled": RunLifecycleStatus.CANCELLED,
            "run_interrupted": RunLifecycleStatus.INTERRUPTED,
            "run_completed": RunLifecycleStatus.COMPLETED,
            "run_failed": RunLifecycleStatus.FAILED,
        }
        terminals = [
            terminal_events[event.event_type]
            for event in events
            if event.event_type in terminal_events
        ]
        if len(terminals) > 1:
            raise ApplicationServiceError("run_terminal_evidence_conflict")
        if terminals and events[-1].event_type not in terminal_events:
            raise ApplicationServiceError("run_event_after_terminal")
        if terminals:
            return terminals[0]
        if any(event.event_type == "run_launched" for event in events):
            return RunLifecycleStatus.RUNNING
        raise ApplicationServiceError("run_lifecycle_incomplete")

    def reconcile_runtime_controller(self, run_id: str) -> ControllerSnapshot:
        """Rebuild non-canonical live ownership from durable worker messages."""

        try:
            require_identifier("run_id", run_id)
        except RecordValidationError:
            raise ApplicationServiceError("run_id_invalid") from None
        run = next(
            (
                stored.record
                for stored in self.store.iter_records()
                if isinstance(stored.record, Run) and stored.record.run_id == run_id
            ),
            None,
        )
        if not isinstance(run, Run):
            raise ApplicationServiceError("run_not_found")
        runtime_request = self._runtime_request_for_identity(run.request_identity)
        messages: list[RuntimeMessage] = []
        for event in self._events(run_id):
            if event.event_type != "run_worker_message":
                continue
            raw = event.payload.get("message")
            if not isinstance(raw, Mapping):
                raise ApplicationServiceError("run_worker_message_invalid")
            try:
                messages.append(RuntimeMessage.from_dict(raw))
            except RuntimeProtocolError:
                raise ApplicationServiceError("run_worker_message_invalid") from None
        if not messages:
            raise ApplicationServiceError("run_worker_messages_missing")
        try:
            controller = SerializedRunController.reconstruct(
                runtime_request.identity, run_id, messages
            )
        except RuntimeControllerError:
            raise ApplicationServiceError("run_worker_reconciliation_failed") from None
        return controller.snapshot()

    def _launch(
        self,
        request: RunLaunchRequest,
        *,
        control: FixtureControl | None,
        attempt_number: int,
        retry_of: Run | None,
        recovery_checkpoint: FixtureCheckpoint | None,
    ) -> RunExecutionResult:
        if not isinstance(request, RunLaunchRequest):
            raise ApplicationServiceError("run_launch_request_invalid")
        claim_identity = content_identity(
            RUN_OWNERSHIP_PROJECTION,
            {
                "schema_version": "v1",
                "run_id": request.run_id,
                "request_id": request.request_id,
                "artifact_id": request.artifact_id,
                "experiment_manifest_identity": identity_fields(
                    request.experiment.manifest_identity
                ),
                "recipe_resolution": record_reference(
                    request.recipe_resolution
                ).to_dict(),
                "dataset_version_identity": identity_fields(
                    request.prepared_dataset.version.identity
                ),
                "execution_target": record_reference(
                    request.execution_target
                ).to_dict(),
                "runtime_identity": identity_fields(self.adapter.runtime_identity),
                "attempt_number": attempt_number,
                "retry_of": (
                    record_reference(retry_of).to_dict()
                    if retry_of is not None
                    else None
                ),
                "recovery_checkpoint_identity": (
                    identity_fields(recovery_checkpoint.checkpoint_identity)
                    if recovery_checkpoint is not None
                    else None
                ),
            },
        )
        try:
            with claim_run_ownership(
                (self.project_root / RUNTIME_OUTPUT_DIRECTORY).absolute(),
                request.run_id,
                claim_identity,
            ) as ownership:
                try:
                    result = self._launch_owned(
                        request,
                        control=control,
                        attempt_number=attempt_number,
                        retry_of=retry_of,
                        recovery_checkpoint=recovery_checkpoint,
                    )
                except Exception:
                    self._resolve_run_ownership(request.run_id, ownership)
                    raise
                self._resolve_run_ownership(request.run_id, ownership)
                return result
        except RunOwnershipError as exc:
            raise ApplicationServiceError(exc.code) from None

    def _resolve_run_ownership(self, run_id: str, ownership: RunOwnershipLease) -> None:
        """Release a launch claim only before launch or after one durable terminal."""

        try:
            events = self._events(run_id)
        except Exception:
            return
        launched = tuple(
            event for event in events if event.event_type == "run_launched"
        )
        if not launched:
            ownership.resolve()
            return
        terminal_types = {
            "run_cancelled",
            "run_interrupted",
            "run_completed",
            "run_failed",
        }
        terminals = tuple(
            event for event in events if event.event_type in terminal_types
        )
        if len(launched) == 1 and len(terminals) == 1 and events[-1] == terminals[0]:
            ownership.resolve()

    def _launch_owned(
        self,
        request: RunLaunchRequest,
        *,
        control: FixtureControl | None,
        attempt_number: int,
        retry_of: Run | None,
        recovery_checkpoint: FixtureCheckpoint | None,
    ) -> RunExecutionResult:
        if not isinstance(request, RunLaunchRequest):
            raise ApplicationServiceError("run_launch_request_invalid")
        if request.evaluation_mode is not EvaluationMode.NO_QUALITY_EVALUATION:
            raise ApplicationServiceError("run_evaluation_mode_not_supported")
        self._assert_unused_run_id(request.run_id)
        self._validate_launch_graph(request)
        try:
            result = preflight(
                request.recipe_resolution,
                request.hardware_requirements,
                request.execution_target,
                request.hardware_capability_profile,
                request.estimate,
            )
        except PreflightError as exc:
            raise ApplicationServiceError(exc.code) from None
        preflight_identity = content_identity(
            PREFLIGHT_EVIDENCE_PROJECTION, result.to_view()
        )
        if not result.ready:
            self._append(
                request.run_id,
                "preflight",
                "run_preflight_blocked",
                {
                    "ready": False,
                    "preflight_identity": identity_fields(preflight_identity),
                    "blocking_reasons": list(result.blocking_reasons),
                },
            )
            raise ApplicationServiceError("run_preflight_blocked")

        try:
            runtime_request, run = self._build_execution_records(
                request,
                preflight_identity,
                attempt_number=attempt_number,
                retry_of=retry_of,
                recovery_checkpoint=recovery_checkpoint,
            )
        except (
            FixtureAdapterError,
            LibraryRuntimeError,
            RecordValidationError,
            TypeError,
            ValueError,
        ):
            raise ApplicationServiceError("run_launch_record_invalid") from None
        start_step = runtime_request.starting_step
        self._persist_launch_records(
            request.hardware_capability_profile, runtime_request, run
        )
        self._append(
            request.run_id,
            "preflight",
            "run_preflight_succeeded",
            {
                "ready": True,
                "preflight_identity": identity_fields(preflight_identity),
                "blocking_reasons": [],
            },
        )
        self._append(
            request.run_id,
            "request-frozen",
            "runtime_request_frozen",
            {
                "runtime_request_identity": identity_fields(runtime_request.identity),
                "experiment_manifest_identity": identity_fields(
                    request.experiment.manifest_identity
                ),
                "preflight_identity": identity_fields(preflight_identity),
                "evaluation_mode": request.evaluation_mode.value,
                "starting_step": start_step,
            },
        )
        if retry_of is not None and recovery_checkpoint is not None:
            self._append(
                request.run_id,
                "recovered",
                "run_recovered",
                {
                    "prior_run_identity": identity_fields(retry_of.identity),
                    "checkpoint_identity": identity_fields(
                        recovery_checkpoint.checkpoint_identity
                    ),
                    "training_state_identity": identity_fields(
                        recovery_checkpoint.training_state_identity
                    ),
                    "starting_step": recovery_checkpoint.step,
                    "resume_compatible": True,
                },
            )
        launched_payload: dict[str, object] = {
            "run_identity": identity_fields(run.identity),
            "runtime_request_identity": identity_fields(runtime_request.identity),
            "attempt_number": attempt_number,
            "fixture_runtime": self.adapter.runtime_kind == "fixture",
        }
        if self.adapter.runtime_kind != "fixture":
            launched_payload["runtime_identity"] = identity_fields(
                self.adapter.runtime_identity
            )
        self._append(
            request.run_id,
            "launched",
            "run_launched",
            launched_payload,
        )
        phase = "runtime"
        try:
            adapter_request = FixtureAdapterRequest(
                request.experiment,
                request.recipe_resolution,
                request.prepared_dataset.version,
                request.prepared_dataset.rendered_bytes,
                runtime_request,
                run,
            )
            if recovery_checkpoint is None:
                output = self.adapter.execute(adapter_request, control=control)
            else:
                output = self.adapter.execute(
                    adapter_request,
                    control=control,
                    resume_checkpoint=recovery_checkpoint,
                )
            phase = "runtime_output"
            self._record_runtime_output(run, output, control)
            if output.termination is FixtureTermination.CANCELLED:
                phase = "cancellation"
                self._append(
                    run.run_id,
                    "cancellation-requested",
                    "run_cancellation_requested",
                    {"acknowledged": True},
                )
                terminal_result = RunExecutionResult(
                    run,
                    runtime_request,
                    result,
                    RunLifecycleStatus.CANCELLED,
                    output.checkpoints,
                )
                self.store.verify()
                self._append(
                    run.run_id,
                    "cancelled",
                    "run_cancelled",
                    {"verified_artifact": False, "terminal": True},
                )
                return terminal_result
            if output.termination is FixtureTermination.INTERRUPTED:
                phase = "interruption"
                terminal_result = RunExecutionResult(
                    run,
                    runtime_request,
                    result,
                    RunLifecycleStatus.INTERRUPTED,
                    output.checkpoints,
                )
                self.store.verify()
                self._append(
                    run.run_id,
                    "interrupted",
                    "run_interrupted",
                    {
                        "verified_artifact": False,
                        "terminal": True,
                        "recovery_checkpoint_count": len(output.checkpoints),
                    },
                )
                return terminal_result
            phase = "artifact_ingestion"
            artifact, availability, integrity = self._ingest_artifact(
                request,
                run,
                runtime_request,
                output,
            )
            phase = "completion"
            terminal_result = RunExecutionResult(
                run,
                runtime_request,
                result,
                RunLifecycleStatus.COMPLETED,
                output.checkpoints,
                artifact,
                availability,
                integrity,
            )
            self.store.verify()
            self._append(
                run.run_id,
                "completed",
                "run_completed",
                {
                    "terminal": True,
                    "verified_artifact": True,
                    "artifact_identity": identity_fields(artifact.identity),
                    "integrity_evidence": identity_fields(integrity.evidence_identity),
                },
            )
            return terminal_result
        except Exception as exc:
            if isinstance(exc, LibraryAdapterExecutionError):
                try:
                    self._record_runtime_boundary_evidence(
                        run,
                        exc.messages,
                        exc.receipts,
                    )
                except Exception as evidence_error:
                    self._terminalize_post_launch_failure(
                        run.run_id, "runtime_output", evidence_error
                    )
            self._terminalize_post_launch_failure(run.run_id, phase, exc)

    def _validate_launch_graph(self, request: RunLaunchRequest) -> None:
        for record in (
            request.experiment,
            request.recipe_resolution,
            request.prepared_dataset.version,
            request.base_model_revision,
            request.compatibility_group,
            request.hardware_requirements,
            request.execution_target,
        ):
            self._require_exact_record(record)
        experiment = request.experiment
        dataset = request.prepared_dataset.version
        resolution = request.recipe_resolution
        if experiment.recipe_resolution != record_reference(resolution):
            raise ApplicationServiceError("run_resolution_mismatch")
        if experiment.dataset_version != dataset.identity:
            raise ApplicationServiceError("run_dataset_mismatch")
        if dataset.tokenizer_identity != request.base_model_revision.tokenizer_identity:
            raise ApplicationServiceError("run_dataset_tokenizer_mismatch")
        if experiment.base_model_revision != record_reference(
            request.base_model_revision
        ):
            raise ApplicationServiceError("run_base_model_mismatch")
        if (
            experiment.tokenizer_identity
            != request.base_model_revision.tokenizer_identity
        ):
            raise ApplicationServiceError("run_tokenizer_mismatch")
        if experiment.compatibility_group != record_reference(
            request.compatibility_group
        ):
            raise ApplicationServiceError("run_compatibility_group_mismatch")
        if experiment.hardware_requirements != record_reference(
            request.hardware_requirements
        ):
            raise ApplicationServiceError("run_hardware_mismatch")
        if experiment.execution_target != record_reference(request.execution_target):
            raise ApplicationServiceError("run_execution_target_mismatch")
        if request.hardware_capability_profile.execution_target != record_reference(
            request.execution_target
        ):
            raise ApplicationServiceError("run_capability_profile_mismatch")
        if resolution.base_model_revision != record_reference(
            request.base_model_revision
        ):
            raise ApplicationServiceError("run_resolution_mismatch")
        if resolution.hardware_requirements != record_reference(
            request.hardware_requirements
        ) or resolution.execution_target != record_reference(request.execution_target):
            raise ApplicationServiceError("run_resolution_mismatch")
        if (
            request.compatibility_group.base_model_revision
            != record_reference(request.base_model_revision)
            or request.compatibility_group.tokenizer_identity
            != request.base_model_revision.tokenizer_identity
            or request.compatibility_group.adapter_type != resolution.adapter_type
            or request.compatibility_group.target_modules != resolution.target_modules
        ):
            raise ApplicationServiceError("run_compatibility_group_mismatch")

    def _build_execution_records(
        self,
        request: RunLaunchRequest,
        preflight_identity: ContentIdentity,
        *,
        attempt_number: int,
        retry_of: Run | None,
        recovery_checkpoint: FixtureCheckpoint | None,
    ) -> tuple[ResolvedRuntimeRequest, Run]:
        start_step = recovery_checkpoint.step if recovery_checkpoint is not None else 0
        training_state = self.adapter.training_state_identity(
            request.experiment,
            request.recipe_resolution,
            request.prepared_dataset.version,
            start_step,
        )
        runtime_request = ResolvedRuntimeRequest(
            request_id=request.request_id,
            experiment=record_reference(request.experiment),
            experiment_manifest_identity=request.experiment.manifest_identity,
            recipe_resolution=record_reference(request.recipe_resolution),
            dataset_version_identity=request.prepared_dataset.version.identity,
            rendered_dataset_identity=(
                request.prepared_dataset.version.rendered_bytes_identity
            ),
            rendered_dataset_byte_count=len(request.prepared_dataset.rendered_bytes),
            hardware_capability_profile=record_reference(
                request.hardware_capability_profile
            ),
            execution_target=record_reference(request.execution_target),
            runtime_identity=self.adapter.runtime_identity,
            preflight_identity=preflight_identity,
            training_state_identity=training_state,
            evaluation_mode=request.evaluation_mode,
            training_steps=request.recipe_resolution.training_steps,
            starting_step=start_step,
            resume_from_run=(
                record_reference(retry_of) if retry_of is not None else None
            ),
            resume_checkpoint_identity=(
                recovery_checkpoint.checkpoint_identity
                if recovery_checkpoint is not None
                else None
            ),
        )
        run = Run(
            run_id=request.run_id,
            experiment=record_reference(request.experiment),
            experiment_manifest_identity=request.experiment.manifest_identity,
            attempt_number=attempt_number,
            hardware_capability_profile=record_reference(
                request.hardware_capability_profile
            ),
            execution_target=record_reference(request.execution_target),
            runtime_identity=self.adapter.runtime_identity,
            request_identity=runtime_request.identity,
            training_state_identity=training_state,
            retry_of=(record_reference(retry_of) if retry_of is not None else None),
        )
        return runtime_request, run

    def _persist_launch_records(
        self,
        profile: HardwareCapabilityProfile,
        runtime_request: ResolvedRuntimeRequest,
        run: Run,
    ) -> None:
        records = (profile, runtime_request, run)
        for record in records:
            require_no_conflicting_logical_revision(
                self.store,
                record,
                conflict_code="run_record_conflict",
            )
        for record in records:
            write_record_idempotently(
                self.store,
                record,
                conflict_code="run_record_conflict",
            )

    def _record_runtime_output(
        self,
        run: Run,
        output: FixtureAdapterOutput,
        control: FixtureControl | None,
    ) -> None:
        del control
        self._record_runtime_boundary_evidence(
            run, output.runtime_messages, output.transfer_receipts
        )
        for progress in output.progress:
            self._append(
                run.run_id,
                f"progress-{progress.step}",
                "run_progress",
                progress.to_dict(),
            )
        for checkpoint in output.checkpoints:
            self._write_idempotent(
                self._checkpoint_path(run, checkpoint), checkpoint.payload
            )
            self._append(
                run.run_id,
                f"checkpoint-{checkpoint.step}",
                "run_checkpoint",
                checkpoint.to_receipt(),
            )
        for log in output.logs:
            self._append(
                run.run_id,
                f"log-{log.ordinal}",
                "run_log",
                log.to_dict(),
            )

    def _record_runtime_boundary_evidence(
        self,
        run: Run,
        messages: tuple[RuntimeMessage, ...],
        receipts: tuple[object, ...],
    ) -> None:
        from temper_ml.runtime.staging import TransferReceipt

        if not isinstance(messages, tuple) or any(
            not isinstance(message, RuntimeMessage)
            or message.run_id != run.run_id
            or message.request_identity != run.request_identity
            for message in messages
        ):
            raise ApplicationServiceError("runtime_worker_message_invalid")
        if not isinstance(receipts, tuple) or any(
            not isinstance(receipt, TransferReceipt) for receipt in receipts
        ):
            raise ApplicationServiceError("runtime_transfer_receipt_invalid")
        typed_receipts = tuple(
            receipt for receipt in receipts if isinstance(receipt, TransferReceipt)
        )
        for message in messages:
            self._append(
                run.run_id,
                f"worker-message-{message.sequence}",
                "run_worker_message",
                {"message": message.to_dict()},
            )
        for index, receipt in enumerate(typed_receipts, 1):
            self._append(
                run.run_id,
                f"transfer-receipt-{index}",
                "runtime_transfer_verified",
                receipt.to_dict(),
            )

    def _ingest_artifact(
        self,
        request: RunLaunchRequest,
        run: Run,
        runtime_request: ResolvedRuntimeRequest,
        output: FixtureAdapterOutput,
    ) -> tuple[Artifact, ArtifactAvailability, ArtifactIntegrityResult]:
        if not output.completed or output.bundle_manifest is None:
            raise ArtifactIntegrityError("artifact_output_incomplete")
        self._append(
            run.run_id,
            "artifact-ingestion-started",
            "artifact_ingestion_started",
            {
                "expected_bundle_identity": identity_fields(
                    output.bundle_manifest.identity
                ),
                "expected_member_count": len(output.bundle_manifest.members),
            },
        )
        root = self._artifact_root(request.artifact_id)
        try:
            ensure_safe_directory(root)
        except (OSError, UnsafeFilesystemPath):
            raise ArtifactIntegrityError("artifact_output_unavailable") from None
        for path, data in output.members.items():
            self._write_idempotent(root / path, data)
        expectation = ArtifactIntegrityExpectation(
            bundle_identity=output.bundle_manifest.identity,
            producing_run=run,
            runtime_request=runtime_request,
            experiment=request.experiment,
            recipe_resolution=request.recipe_resolution,
            dataset_version=request.prepared_dataset.version,
            base_model_revision=request.base_model_revision,
            compatibility_group=request.compatibility_group,
        )
        integrity = verify_artifact_bundle(root, expectation)
        try:
            self.store.write_bundle_manifest(integrity.bundle_manifest)
        except EvidenceExists:
            if (
                self.store.read_bundle_manifest(integrity.bundle_manifest.identity)
                != integrity.bundle_manifest
            ):
                raise ArtifactIntegrityError("artifact_manifest_store_mismatch")
        lineage = content_identity(
            ARTIFACT_LINEAGE_PROJECTION,
            {
                "schema_version": "v1",
                "experiment": record_reference(request.experiment).to_dict(),
                "resolved_runtime_request": record_reference(runtime_request).to_dict(),
                "producing_run": record_reference(run).to_dict(),
            },
        )
        artifact = Artifact(
            artifact_id=request.artifact_id,
            project=request.experiment.project,
            producing_run=record_reference(run),
            adapter_type=request.recipe_resolution.adapter_type,
            content_kind=ArtifactContentKind.BUNDLE,
            content_identity=integrity.bundle_manifest.identity,
            base_model_revision=record_reference(request.base_model_revision),
            tokenizer_identity=request.base_model_revision.tokenizer_identity,
            compatibility_groups=(record_reference(request.compatibility_group),),
            parent_artifacts=(),
            storage_references=(
                StorageReference("project_store", request.artifact_id),
            ),
            integrity_evidence=integrity.evidence_identity,
            provenance=integrity.provenance_identity,
            lineage_evidence=lineage,
        )
        availability = ArtifactAvailability(
            availability_id=f"available-{artifact.identity.value[:24]}",
            artifact=record_reference(artifact),
            state=AvailabilityState.AVAILABLE,
            available_byte_classes=("final_adapter",),
            storage_references=artifact.storage_references,
            checkpoint_resumable=False,
            observed_content_identity=artifact.content_identity,
        )
        for record in (artifact, availability):
            require_no_conflicting_logical_revision(
                self.store,
                record,
                conflict_code="artifact_record_conflict",
            )
            write_record_idempotently(
                self.store,
                record,
                conflict_code="artifact_record_conflict",
            )
        self._append(
            run.run_id,
            "artifact-ingestion-verified",
            "artifact_ingestion_verified",
            {
                "artifact_identity": identity_fields(artifact.identity),
                "bundle_identity": identity_fields(artifact.content_identity),
                "integrity_evidence": identity_fields(integrity.evidence_identity),
                "quality_evaluation_required": False,
            },
        )
        return artifact, availability, integrity

    def _assert_unused_run_id(self, run_id: str) -> None:
        if self._events(run_id) or any(
            isinstance(stored.record, Run) and stored.record.run_id == run_id
            for stored in self.store.iter_records()
        ):
            raise ApplicationServiceError("run_id_already_used")

    def _require_exact_record(self, record: TypedRecord) -> TypedRecord:
        try:
            stored = self.store.read_record(record_reference(record))
        except (EvidenceError, RecordValidationError):
            raise ApplicationServiceError("run_dependency_missing") from None
        if (
            type(stored.record) is not type(record)
            or stored.envelope.to_dict() != record.to_dict()
        ):
            raise ApplicationServiceError("run_dependency_mismatch")
        return stored.record

    def _runtime_request_for_identity(
        self, identity: ContentIdentity
    ) -> ResolvedRuntimeRequest:
        matches = [
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, ResolvedRuntimeRequest)
            and stored.record.identity == identity
        ]
        if len(matches) != 1:
            raise ApplicationServiceError("run_runtime_request_missing")
        return matches[0]

    def _checkpoint_event(self, run_id: str, identity: ContentIdentity) -> StoredEvent:
        events = self._events(run_id)
        matches = [
            event
            for event in events
            if event.event_type == "run_checkpoint"
            and _event_identity(event, "checkpoint_identity") == identity
        ]
        if len(matches) != 1:
            raise ApplicationServiceError("run_recovery_checkpoint_not_found")
        return matches[0]

    def _checkpoint_resume_available(
        self,
        run_id: str,
        checkpoint: StoredEvent,
    ) -> bool:
        identity = _event_identity(checkpoint, "checkpoint_identity")
        resume_available = checkpoint.payload.get("resume_compatible") is True
        for event in self._events(run_id):
            if event.event_type not in {
                "run_checkpoint_cleanup_pending",
                "run_checkpoint_cleanup_cancelled",
                "run_checkpoint_removed",
            }:
                continue
            if _event_identity(event, "content_identity") != identity:
                continue
            resume_available = (
                event.event_type == "run_checkpoint_cleanup_cancelled"
                and event.payload.get("resume_available") is True
            )
        return resume_available

    def _checkpoint_from_event(self, run: Run, event: StoredEvent) -> FixtureCheckpoint:
        if event.event_type != "run_checkpoint":
            raise ApplicationServiceError("run_event_invalid")
        resume_compatible = event.payload.get("resume_compatible")
        if not isinstance(resume_compatible, bool):
            raise ApplicationServiceError("run_existing_checkpoint_conflict")
        step = _event_positive_int(event, "step")
        runtime_request = self._runtime_request_for_identity(run.request_identity)
        if resume_compatible != (step < runtime_request.training_steps):
            raise ApplicationServiceError("run_existing_checkpoint_conflict")
        state_identity = _event_identity(event, "training_state_identity")
        checkpoint_identity = _event_identity(event, "checkpoint_identity")
        provisional = FixtureCheckpoint(
            step,
            state_identity,
            checkpoint_identity,
            b"",
            resume_compatible,
        )
        try:
            payload = read_stable_bytes(self._checkpoint_path(run, provisional))
        except SafeIoError:
            raise ApplicationServiceError("run_existing_checkpoint_conflict") from None
        checkpoint = FixtureCheckpoint(
            step,
            state_identity,
            checkpoint_identity,
            payload,
            resume_compatible,
        )
        if (
            len(payload) != _event_positive_int(event, "byte_count")
            or ContentIdentity("sha256", hashlib.sha256(payload).hexdigest())
            != checkpoint_identity
        ):
            raise ApplicationServiceError("run_existing_checkpoint_conflict")
        return checkpoint

    def _events(self, run_id: str) -> tuple[StoredEvent, ...]:
        stream_id = self._stream_id(run_id)
        return next(
            (
                snapshot.events
                for snapshot in self.store.iter_streams()
                if snapshot.stream_id == stream_id
            ),
            (),
        )

    def _append(
        self,
        run_id: str,
        key: str,
        event_type: str,
        payload: Mapping[str, object],
    ) -> StoredEvent:
        request = EventRequest(f"{run_id}-{key}", event_type, payload)
        try:
            return self.store.append_event(self._stream_id(run_id), request)
        except Exception:
            reconciled = self._reconcile_ambiguous_append(run_id, request)
            if reconciled is not None:
                return reconciled
            raise

    def _reconcile_ambiguous_append(
        self, run_id: str, request: EventRequest
    ) -> StoredEvent | None:
        try:
            matches = tuple(
                event
                for event in self._events(run_id)
                if event.idempotency_key == request.idempotency_key
            )
        except Exception:
            return None
        if len(matches) != 1:
            return None
        durable = matches[0]
        if durable.request_fields() != request.canonical_fields():
            return None
        return durable

    def _append_failure(self, run_id: str, phase: str, code: str) -> None:
        self._append(
            run_id,
            "failed",
            "run_failed",
            {
                "terminal": True,
                "phase": phase,
                "failure_code": code,
                "verified_artifact": False,
            },
        )

    def _terminalize_post_launch_failure(
        self,
        run_id: str,
        phase: str,
        error: Exception,
    ) -> NoReturn:
        stable_errors = (
            ApplicationServiceError,
            ArtifactIntegrityError,
            EvidenceError,
            FixtureAdapterError,
            LibraryRuntimeError,
        )
        code = error.code if isinstance(error, stable_errors) else None
        if not isinstance(code, str) or not code:
            code = {
                "runtime": "run_runtime_failed",
                "runtime_output": "run_output_persistence_failed",
                "cancellation": "run_cancellation_persistence_failed",
                "interruption": "run_interruption_persistence_failed",
                "artifact_ingestion": "artifact_ingestion_failed",
                "completion": "run_completion_persistence_failed",
            }.get(phase, "run_post_launch_failed")
        if phase == "artifact_ingestion":
            try:
                self._append(
                    run_id,
                    "artifact-ingestion-failed",
                    "artifact_ingestion_failed",
                    {"failure_code": code, "verified_artifact": False},
                )
            except EvidenceError:
                pass
        self._append_failure(run_id, phase, code)
        raise ApplicationServiceError(code) from None

    def _stream_id(self, run_id: str) -> str:
        return f"run-{run_id}"

    def _runtime_root(self) -> Path:
        return self.project_root / RUNTIME_OUTPUT_DIRECTORY

    def _artifact_root(self, artifact_id: str) -> Path:
        return self._runtime_root() / "artifacts" / artifact_id

    def _checkpoint_path(self, run: Run, checkpoint: FixtureCheckpoint) -> Path:
        return (
            self._runtime_root()
            / "checkpoints"
            / run.run_id
            / (f"{checkpoint.step:08d}-{checkpoint.checkpoint_identity.value}.json")
        )

    @staticmethod
    def _write_idempotent(path: Path, data: bytes) -> None:
        try:
            write_once_bytes(path, data)
        except FileExistsError:
            try:
                existing = read_stable_bytes(path)
            except SafeIoError:
                raise SafeIoError("existing runtime output is unreadable") from None
            if existing != data:
                raise SafeIoError("existing runtime output differs")


def _event_identity(event: StoredEvent, field: str) -> ContentIdentity:
    raw = event.payload.get(field)
    if not isinstance(raw, Mapping):
        raise ApplicationServiceError("run_event_invalid")
    try:
        return parse_identity(raw, field=field)
    except RecordValidationError:
        raise ApplicationServiceError("run_event_invalid") from None


def _event_positive_int(event: StoredEvent, field: str) -> int:
    value = event.payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ApplicationServiceError("run_event_invalid")
    return value
