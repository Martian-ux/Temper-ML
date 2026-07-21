from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import pickle
import shutil
from threading import Event

import pytest
import temper_ml.app_services.runs as runs_module
import temper_ml.runtime.library_backend as library_backend_module

from temper_ml.app_services.datasets import (
    DatasetImportRequest,
    DatasetService,
    PreparedDataset,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.local_use import (
    AdapterExportRequest,
    LocalUseRequest,
    LocalUseService,
)
from temper_ml.app_services.runs import (
    RunLaunchRequest,
    RunLifecycleStatus,
    RunRecoveryRequest,
    RunService,
)
from temper_ml.app_services.retention import (
    ByteClass,
    CleanupImpact,
    RetentionService,
)
from temper_ml.cli import _FixtureTokenizer, _fixture_workflow
from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactAvailability,
    AvailabilityState,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import CompatibilityGroup
from temper_ml.domain.datasets import (
    DeduplicationRule,
    FieldMapping,
    FilterRule,
    RendererSpec,
    SplitPart,
    SplitRule,
)
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.local_use import LocalUseSession
from temper_ml.domain.projections import ContentIdentity, content_identity
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import identity_fields, record_reference
from temper_ml.domain.retention import CleanupObjectStatus, CleanupOutcome
from temper_ml.domain.runs import EvaluationMode, ResolvedRuntimeRequest, Run
from temper_ml.runtime.fixture_adapter import (
    FixtureAdapter,
    FixtureAdapterError,
    FixtureAdapterRequest,
    FixtureControl,
)
from temper_ml.runtime.controller import ControllerState
from temper_ml.runtime.fixture_inference import (
    FixtureInferenceRequest,
    InferenceSettings,
)
from temper_ml.runtime.library_adapter import (
    LibraryAdapter,
    LibraryInferenceRuntime,
    LibraryRuntimeSources,
)
from temper_ml.runtime.library_backend import (
    LibraryCapability,
    LibraryRuntimeError,
    LibraryTrainingResult,
)
from temper_ml.runtime.library_double import DeterministicLibraryBackend
from temper_ml.runtime.ownership import (
    RunOwnershipError,
    RunOwnershipLease,
    claim_run_ownership,
    released_run_claim_identity,
)
from temper_ml.runtime.preflight import EstimateComponents, estimate_resources
from temper_ml.runtime.protocol import RuntimeOperation
from temper_ml.store.canonical_json import (
    dumps_canonical_json,
    loads_canonical_json,
)
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest
from temper_ml.store.safe_io import SafeIoError


@dataclass(frozen=True)
class _Foundation:
    prepared: PreparedDataset
    experiment: Experiment
    resolution: RecipeResolution
    model: BaseModelRevision
    group: CompatibilityGroup
    requirements: HardwareRequirements
    target: ExecutionTarget
    profile: HardwareCapabilityProfile

    def launch(self, *, run_id: str, request_id: str, artifact_id: str):
        estimate = estimate_resources(
            self.resolution,
            EstimateComponents(
                0,
                0,
                0,
                0,
                len(self.prepared.rendered_bytes),
                1024,
            ),
        )
        return RunLaunchRequest(
            run_id,
            request_id,
            artifact_id,
            self.experiment,
            self.resolution,
            self.prepared,
            self.model,
            self.group,
            self.requirements,
            self.target,
            self.profile,
            estimate,
            EvaluationMode.NO_QUALITY_EVALUATION,
        )


def _one(store: TypedEvidenceStore, kind):
    values = [
        record.record
        for record in store.iter_records()
        if isinstance(record.record, kind)
    ]
    assert len(values) == 1
    return values[0]


def _project_file_snapshot(root: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        )
    )


def _copy_with_run_events(
    source: Path,
    target: Path,
    run_id: str,
    events,
) -> None:
    shutil.copytree(source, target)
    store = TypedEvidenceStore(target)
    stream_id = f"run-{run_id}"
    shutil.rmtree(store.layout.stream_events(stream_id).parent)
    for event in events:
        fields = event.request_fields()
        store.append_event(
            stream_id,
            EventRequest(
                event.idempotency_key,
                event.event_type,
                fields["payload"],
            ),
        )


def _foundation(root: Path) -> _Foundation:
    _fixture_workflow(str(root), evaluation_mode=EvaluationMode.NO_QUALITY_EVALUATION)
    store = TypedEvidenceStore(root)
    rows = [
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
    prepared = DatasetService(root).import_json(
        dumps_canonical_json(rows),
        DatasetImportRequest(
            "dataset-fixture-runtime",
            FieldMapping("instruction", "response", "context"),
            RendererSpec(),
            FilterRule(1, 1000, 1000),
            DeduplicationRule(),
            SplitRule(17, (SplitPart("train", 4), SplitPart("validation", 1))),
            _FixtureTokenizer(),
            2,
        ),
    )
    return _Foundation(
        prepared,
        _one(store, Experiment),
        _one(store, RecipeResolution),
        _one(store, BaseModelRevision),
        _one(store, CompatibilityGroup),
        _one(store, HardwareRequirements),
        _one(store, ExecutionTarget),
        _one(store, HardwareCapabilityProfile),
    )


def _library_foundation(
    root: Path,
) -> tuple[
    _Foundation,
    DeterministicLibraryBackend,
    LibraryRuntimeSources,
    LibraryAdapter,
]:
    fixture = _foundation(root)
    versions = {
        "accelerate": "1.test",
        "peft": "1.test",
        "torch": "1.test",
        "transformers": "1.test",
    }
    capability = LibraryCapability(
        accelerator_backend=fixture.target.accelerator_backend,
        accelerator_architecture="synthetic-library-cpu",
        accelerator_model="Synthetic library CPU",
        accelerator_count=0,
        accelerator_memory_bytes=(),
        system_memory_bytes=1_000_000,
        supported_precision_modes=("fp32",),
        supported_quantization_modes=("none",),
        capabilities=(
            "accelerate",
            "cancellation",
            "checkpoint_resume",
            "evaluation_inference",
            "fixture_adapter",
            "local_staging",
            "local_use_inference",
            "lora",
            "peft",
            "transformers",
        ),
        library_versions=versions,
    )
    backend = DeterministicLibraryBackend(capability)
    sources = LibraryRuntimeSources(
        (root.resolve() / "private-model-cache"),
        (root.resolve() / "private-tokenizer-cache"),
        (root.resolve() / ".temper-fixture-output" / "library-staging"),
        fixture.target.target_class,
        record_reference(fixture.model),
        fixture.model.tokenizer_identity,
    )
    adapter = LibraryAdapter(backend, sources, capability=capability)
    resolution = replace(
        fixture.resolution,
        resolution_id="resolution-library-runtime",
        library_versions=versions,
    )
    experiment = replace(
        fixture.experiment,
        experiment_id="experiment-library-runtime",
        recipe_resolution=record_reference(resolution),
    )
    profile = adapter.capability_profile("profile-library-runtime", fixture.target)
    store = TypedEvidenceStore(root)
    for record in (resolution, experiment, profile):
        store.write_record(record)
    return (
        replace(
            fixture,
            experiment=experiment,
            resolution=resolution,
            profile=profile,
        ),
        backend,
        sources,
        adapter,
    )


def test_cancellation_is_terminal_and_has_no_verified_artifact(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)

    result = service.launch(
        foundation.launch(
            run_id="run-cancelled",
            request_id="request-cancelled",
            artifact_id="artifact-cancelled",
        ),
        control=FixtureControl(cancel_after_step=2),
    )

    assert result.status is RunLifecycleStatus.CANCELLED
    assert result.artifact is None
    assert result.integrity is None
    assert service.status(result.run.run_id) is RunLifecycleStatus.CANCELLED
    assert not any(
        isinstance(stored.record, Artifact)
        and stored.record.artifact_id == "artifact-cancelled"
        for stored in service.store.iter_records()
    )
    event_types = [event.event_type for event in service._events(result.run.run_id)]
    assert "run_cancellation_requested" in event_types
    assert event_types[-1] == "run_cancelled"


def test_interruption_recovery_creates_new_attempt_from_bound_checkpoint(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)
    interrupted = service.launch(
        foundation.launch(
            run_id="run-interrupted",
            request_id="request-interrupted",
            artifact_id="artifact-interrupted",
        ),
        control=FixtureControl(interrupt_after_step=3),
    )
    before = tuple(service._events(interrupted.run.run_id))

    recovered = service.recover(
        RunRecoveryRequest(
            foundation.launch(
                run_id="run-recovered",
                request_id="request-recovered",
                artifact_id="artifact-recovered",
            ),
            interrupted.run,
            interrupted.checkpoints[-1].checkpoint_identity,
        )
    )

    assert interrupted.status is RunLifecycleStatus.INTERRUPTED
    assert recovered.status is RunLifecycleStatus.COMPLETED
    assert recovered.run.attempt_number == 2
    assert recovered.run.retry_of is not None
    assert recovered.runtime_request.starting_step == 3
    assert recovered.runtime_request.resume_checkpoint_identity == (
        interrupted.checkpoints[-1].checkpoint_identity
    )
    assert tuple(service._events(interrupted.run.run_id)) == before
    assert service.status(recovered.run.run_id) is RunLifecycleStatus.COMPLETED
    assert recovered.artifact is not None


def test_failed_checkpoint_cleanup_retains_real_recovery(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    interrupted = RunService(tmp_path).launch(
        foundation.launch(
            run_id="run-cleanup-retained",
            request_id="request-cleanup-retained",
            artifact_id="artifact-cleanup-retained",
        ),
        control=FixtureControl(interrupt_after_step=3),
    )
    checkpoint = interrupted.checkpoints[-1]

    def retain_file(path: Path) -> None:
        raise PermissionError(path.name)

    retention = RetentionService(tmp_path, _remove_file=retain_file)
    entry = next(
        item
        for item in retention.inventory().entries
        if item.byte_class is ByteClass.CHECKPOINT
        and CleanupImpact.RESUMABILITY in item.impacts
        and item.content_identity == checkpoint.checkpoint_identity
        and any(
            subject.record_type == "run"
            and subject.logical_id == interrupted.run.run_id
            for subject in item.subjects
        )
    )
    receipt = retention.execute(
        retention.plan((entry.entry_id,)),
        confirm=True,
    )

    assert receipt.outcome is CleanupOutcome.FAILED
    assert receipt.objects[0].status is CleanupObjectStatus.RETAINED
    assert entry._path.exists()
    recovered = RunService(tmp_path).recover(
        RunRecoveryRequest(
            foundation.launch(
                run_id="run-cleanup-retained-recovery",
                request_id="request-cleanup-retained-recovery",
                artifact_id="artifact-cleanup-retained-recovery",
            ),
            interrupted.run,
            checkpoint.checkpoint_identity,
        )
    )
    assert recovered.status is RunLifecycleStatus.COMPLETED
    assert recovered.runtime_request.starting_step == checkpoint.step


def test_removed_checkpoint_does_not_block_recovery_from_another(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    resolution = replace(
        foundation.resolution,
        resolution_id="resolution-multiple-recovery-checkpoints",
        training_steps=8,
        checkpoint_cadence=2,
    )
    experiment = replace(
        foundation.experiment,
        experiment_id="experiment-multiple-recovery-checkpoints",
        recipe_resolution=record_reference(resolution),
    )
    store = TypedEvidenceStore(tmp_path)
    store.write_record(resolution)
    store.write_record(experiment)
    foundation = replace(
        foundation,
        experiment=experiment,
        resolution=resolution,
    )
    interrupted = RunService(tmp_path).launch(
        foundation.launch(
            run_id="run-multiple-recovery-checkpoints",
            request_id="request-multiple-recovery-checkpoints",
            artifact_id="artifact-multiple-recovery-checkpoints",
        ),
        control=FixtureControl(interrupt_after_step=5),
    )
    assert len(interrupted.checkpoints) >= 2
    removed_checkpoint = interrupted.checkpoints[0]
    retained_checkpoint = interrupted.checkpoints[-1]
    retention = RetentionService(tmp_path)
    removed_entry = next(
        item
        for item in retention.inventory().entries
        if item.byte_class is ByteClass.CHECKPOINT
        and item.content_identity == removed_checkpoint.checkpoint_identity
        and any(
            subject.record_type == "run"
            and subject.logical_id == interrupted.run.run_id
            for subject in item.subjects
        )
    )

    receipt = retention.execute(
        retention.plan((removed_entry.entry_id,)),
        confirm=True,
    )

    assert receipt.outcome is CleanupOutcome.COMPLETED
    assert not removed_entry._path.exists()
    recovered = RunService(tmp_path).recover(
        RunRecoveryRequest(
            foundation.launch(
                run_id="run-retained-checkpoint-recovery",
                request_id="request-retained-checkpoint-recovery",
                artifact_id="artifact-retained-checkpoint-recovery",
            ),
            interrupted.run,
            retained_checkpoint.checkpoint_identity,
        )
    )
    assert recovered.status is RunLifecycleStatus.COMPLETED
    assert recovered.runtime_request.starting_step == retained_checkpoint.step


def test_recovery_rejects_unrecorded_checkpoint_identity(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)
    interrupted = service.launch(
        foundation.launch(
            run_id="run-interrupted-invalid",
            request_id="request-interrupted-invalid",
            artifact_id="artifact-interrupted-invalid",
        ),
        control=FixtureControl(interrupt_after_step=3),
    )

    with pytest.raises(
        ApplicationServiceError, match="run_recovery_checkpoint_not_found"
    ):
        service.recover(
            RunRecoveryRequest(
                foundation.launch(
                    run_id="run-recovery-invalid",
                    request_id="request-recovery-invalid",
                    artifact_id="artifact-recovery-invalid",
                ),
                interrupted.run,
                ContentIdentity("sha256", "f" * 64),
            )
        )


def test_completed_run_reopens_exactly_and_conflicting_request_fails_closed(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)
    launch = foundation.launch(
        run_id="run-reopen-completed",
        request_id="request-reopen-completed",
        artifact_id="artifact-reopen-completed",
    )
    completed = service.launch(launch)
    before = service._events(launch.run_id)
    before_files = _project_file_snapshot(tmp_path)

    reopened = RunService(tmp_path, adapter=_FailingAdapter()).reopen_completed(launch)

    assert reopened == completed
    assert service._events(launch.run_id) == before
    assert _project_file_snapshot(tmp_path) == before_files
    with pytest.raises(ApplicationServiceError, match="run_existing_conflict"):
        service.reopen_completed(
            replace(launch, request_id="request-reopen-conflicting")
        )


def test_completed_run_reopens_with_retained_checkpoint_cleanup_observations(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-reopen-after-cleanup",
        request_id="request-reopen-after-cleanup",
        artifact_id="artifact-reopen-after-cleanup",
    )
    completed = RunService(tmp_path).launch(launch)
    checkpoint = completed.checkpoints[0]

    def retain_file(path: Path) -> None:
        raise PermissionError(path.name)

    retention = RetentionService(tmp_path, _remove_file=retain_file)
    entry = next(
        item
        for item in retention.inventory().entries
        if item.byte_class is ByteClass.CHECKPOINT
        and item.content_identity == checkpoint.checkpoint_identity
        and any(
            subject.record_type == "run" and subject.logical_id == completed.run.run_id
            for subject in item.subjects
        )
    )
    receipt = retention.execute(
        retention.plan((entry.entry_id,)),
        confirm=True,
    )

    assert receipt.objects[0].status is CleanupObjectStatus.RETAINED
    reopened = RunService(tmp_path).reopen_completed(launch)
    assert reopened == completed
    assert (
        RunService(tmp_path).status(completed.run.run_id)
        is RunLifecycleStatus.COMPLETED
    )


@pytest.mark.parametrize("mutation", ["missing", "reordered"])
def test_reopen_completed_requires_the_exact_success_lifecycle(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source"
    copied = tmp_path / mutation
    foundation = _foundation(source)
    service = RunService(source)
    launch = foundation.launch(
        run_id=f"run-lifecycle-{mutation}",
        request_id=f"request-lifecycle-{mutation}",
        artifact_id=f"artifact-lifecycle-{mutation}",
    )
    completed = service.launch(launch)
    events = list(service._events(launch.run_id))
    progress_indices = [
        index
        for index, event in enumerate(events)
        if event.event_type == "run_progress"
    ]
    assert len(progress_indices) >= 2
    if mutation == "missing":
        events.pop(progress_indices[0])
    else:
        first, second = progress_indices[:2]
        events[first], events[second] = events[second], events[first]
    _copy_with_run_events(source, copied, launch.run_id, events)
    before = _project_file_snapshot(copied)

    with pytest.raises(
        ApplicationServiceError, match="run_existing_lifecycle_conflict"
    ):
        RunService(copied).reopen_completed(launch)

    assert _project_file_snapshot(copied) == before
    assert service.reopen_completed(launch) == completed


def test_reopen_completed_requires_the_stored_bundle_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    copied = tmp_path / "missing-manifest"
    foundation = _foundation(source)
    service = RunService(source)
    launch = foundation.launch(
        run_id="run-manifest-missing",
        request_id="request-manifest-missing",
        artifact_id="artifact-manifest-missing",
    )
    completed = service.launch(launch)
    assert completed.integrity is not None
    shutil.copytree(source, copied)
    copied_store = TypedEvidenceStore(copied)
    copied_store.layout.bundle_manifest_path(
        completed.integrity.bundle_manifest.identity
    ).unlink()
    before = _project_file_snapshot(copied)

    with pytest.raises(ApplicationServiceError, match="run_existing_artifact_conflict"):
        RunService(copied).reopen_completed(launch)

    assert _project_file_snapshot(copied) == before
    assert service.reopen_completed(launch) == completed


def test_failed_run_cannot_be_reopened_as_completed(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path, adapter=_FailingAdapter())
    launch = foundation.launch(
        run_id="run-reopen-failed",
        request_id="request-reopen-failed",
        artifact_id="artifact-reopen-failed",
    )

    with pytest.raises(ApplicationServiceError, match="fixture_failure_injected"):
        service.launch(launch)
    with pytest.raises(ApplicationServiceError, match="run_existing_not_completed"):
        service.reopen_completed(launch)


def test_preflight_blocks_before_request_freeze_or_launch(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-preflight-blocked",
        request_id="request-preflight-blocked",
        artifact_id="artifact-preflight-blocked",
    )
    blocked = replace(
        launch, estimate=replace(launch.estimate, accelerator_memory_bytes=1)
    )
    service = RunService(tmp_path)

    with pytest.raises(ApplicationServiceError, match="run_preflight_blocked"):
        service.launch(blocked)

    assert service.status(blocked.run_id) is RunLifecycleStatus.PREFLIGHT_BLOCKED
    assert [event.event_type for event in service._events(blocked.run_id)] == [
        "run_preflight_blocked"
    ]
    assert not any(
        isinstance(stored.record, Run) and stored.record.run_id == blocked.run_id
        for stored in service.store.iter_records()
    )


def test_startup_releases_event_only_preflight_blocked_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-preflight-restart",
        request_id="request-preflight-restart",
        artifact_id="artifact-preflight-restart",
    )
    blocked = replace(
        launch, estimate=replace(launch.estimate, accelerator_memory_bytes=1)
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(blocked)

    def fail_resolution(_lease: RunOwnershipLease) -> None:
        raise RunOwnershipError("run_ownership_resolution_failed")

    with monkeypatch.context() as resolution_failure:
        resolution_failure.setattr(RunOwnershipLease, "resolve", fail_resolution)
        with pytest.raises(
            ApplicationServiceError, match="^run_ownership_resolution_failed$"
        ):
            service.launch(blocked)

    assert service.status(blocked.run_id) is RunLifecycleStatus.PREFLIGHT_BLOCKED
    assert RunService(tmp_path).reconcile_abandoned_runs() == (blocked.run_id,)
    assert (
        released_run_claim_identity(
            (tmp_path / ".temper-fixture-output").resolve(), blocked.run_id
        )
        == claim
    )
    assert service.status(blocked.run_id) is RunLifecycleStatus.PREFLIGHT_BLOCKED


@pytest.mark.parametrize("lost_after", [ResolvedRuntimeRequest, Run])
def test_startup_terminalizes_claim_bound_prelaunch_record_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lost_after: type[ResolvedRuntimeRequest] | type[Run],
) -> None:
    foundation = _foundation(tmp_path)
    suffix = lost_after.RECORD_TYPE
    launch = foundation.launch(
        run_id=f"run-prelaunch-loss-{suffix}",
        request_id=f"request-prelaunch-loss-{suffix}",
        artifact_id=f"artifact-prelaunch-loss-{suffix}",
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(launch)
    original_write = runs_module.write_record_idempotently
    lost = False

    def write_then_lose_process(*args: object, **kwargs: object) -> None:
        nonlocal lost
        original_write(*args, **kwargs)  # type: ignore[arg-type]
        record = args[1] if len(args) > 1 else kwargs.get("record")
        if isinstance(record, lost_after) and not lost:
            lost = True
            raise SystemExit("synthetic process loss during launch records")

    with monkeypatch.context() as process_loss:
        process_loss.setattr(
            runs_module, "write_record_idempotently", write_then_lose_process
        )
        with pytest.raises(SystemExit, match="during launch records"):
            service.launch(launch)

    assert lost is True
    assert service._events(launch.run_id) == ()
    restarted = RunService(tmp_path)
    assert restarted.reconcile_abandoned_runs() == (launch.run_id,)
    assert restarted.status(launch.run_id) is RunLifecycleStatus.FAILED
    run = next(
        stored.record
        for stored in restarted.store.iter_records()
        if isinstance(stored.record, Run) and stored.record.run_id == launch.run_id
    )
    assert run.request_identity == next(
        stored.record.identity
        for stored in restarted.store.iter_records()
        if isinstance(stored.record, ResolvedRuntimeRequest)
        and stored.record.request_id == launch.request_id
    )
    terminal = restarted._events(launch.run_id)[-1]
    assert terminal.event_type == "run_failed"
    assert terminal.payload.get("phase") == "launch_records"
    assert (
        terminal.payload.get("failure_code") == "run_launch_record_persistence_failed"
    )
    assert (
        released_run_claim_identity(
            (tmp_path / ".temper-fixture-output").resolve(), launch.run_id
        )
        == claim
    )


def test_launch_record_failure_never_terminalizes_without_a_durable_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-record-write-before-commit",
        request_id="request-record-write-before-commit",
        artifact_id="artifact-record-write-before-commit",
    )
    service = RunService(tmp_path)
    original_write = runs_module.write_record_idempotently

    def fail_run_before_commit(*args: object, **kwargs: object) -> None:
        record = args[1] if len(args) > 1 else kwargs.get("record")
        if isinstance(record, Run) and record.run_id == launch.run_id:
            raise EvidenceError("synthetic_run_record_write_failed")
        original_write(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as record_failure:
        record_failure.setattr(
            runs_module, "write_record_idempotently", fail_run_before_commit
        )
        with pytest.raises(
            ApplicationServiceError, match="^synthetic_run_record_write_failed$"
        ):
            service.launch(launch)

    assert service._events(launch.run_id) == ()
    assert not any(
        isinstance(stored.record, Run) and stored.record.run_id == launch.run_id
        for stored in service.store.iter_records()
    )
    assert any(
        isinstance(stored.record, ResolvedRuntimeRequest)
        and stored.record.request_id == launch.request_id
        for stored in service.store.iter_records()
    )

    restarted = RunService(tmp_path)
    assert restarted.reconcile_abandoned_runs() == (launch.run_id,)
    assert restarted.status(launch.run_id) is RunLifecycleStatus.FAILED
    assert (
        sum(
            event.event_type == "run_failed"
            for event in restarted._events(launch.run_id)
        )
        == 1
    )
    assert (
        sum(
            isinstance(stored.record, Run) and stored.record.run_id == launch.run_id
            for stored in restarted.store.iter_records()
        )
        == 1
    )


def test_prelaunch_recovery_releases_claim_after_second_process_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-prelaunch-recovery-second-loss",
        request_id="request-prelaunch-recovery-second-loss",
        artifact_id="artifact-prelaunch-recovery-second-loss",
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(launch)
    original_write = runs_module.write_record_idempotently

    def write_request_then_lose_process(*args: object, **kwargs: object) -> None:
        original_write(*args, **kwargs)  # type: ignore[arg-type]
        record = args[1] if len(args) > 1 else kwargs.get("record")
        if (
            isinstance(record, ResolvedRuntimeRequest)
            and record.request_id == launch.request_id
        ):
            raise SystemExit("synthetic first prelaunch process loss")

    with monkeypatch.context() as first_loss:
        first_loss.setattr(
            runs_module, "write_record_idempotently", write_request_then_lose_process
        )
        with pytest.raises(SystemExit, match="first prelaunch process loss"):
            service.launch(launch)

    original_append = RunService._append

    def append_failure_then_lose_process(
        run_service: RunService,
        run_id: str,
        key: str,
        event_type: str,
        payload: object,
    ) -> object:
        event = original_append(
            run_service,
            run_id,
            key,
            event_type,
            payload,  # type: ignore[arg-type]
        )
        if run_id == launch.run_id and event_type == "run_failed":
            raise SystemExit("synthetic second prelaunch process loss")
        return event

    with monkeypatch.context() as second_loss:
        second_loss.setattr(RunService, "_append", append_failure_then_lose_process)
        with pytest.raises(SystemExit, match="second prelaunch process loss"):
            RunService(tmp_path).reconcile_abandoned_runs()

    root = (tmp_path / ".temper-fixture-output").resolve()
    after_second_loss = RunService(tmp_path)
    assert after_second_loss.status(launch.run_id) is RunLifecycleStatus.FAILED
    with pytest.raises(RunOwnershipError, match="^run_ownership_unresolved$"):
        released_run_claim_identity(root, launch.run_id)

    assert after_second_loss.reconcile_abandoned_runs() == (launch.run_id,)
    assert released_run_claim_identity(root, launch.run_id) == claim
    assert (
        sum(
            event.event_type == "run_failed"
            for event in after_second_loss._events(launch.run_id)
        )
        == 1
    )


def _persist_synthetic_running_attempt(
    service: RunService,
    launch: RunLaunchRequest,
    claim: ContentIdentity,
    *,
    legacy_ownership_evidence: bool = False,
) -> None:
    preflight_result = runs_module.preflight(
        launch.recipe_resolution,
        launch.hardware_requirements,
        launch.execution_target,
        launch.hardware_capability_profile,
        launch.estimate,
    )
    runtime_request, run = service.planned_first_attempt(launch, preflight_result)
    service._persist_launch_records(
        launch.hardware_capability_profile,
        runtime_request,
        run,
    )
    payload: dict[str, object] = {
        "run_identity": identity_fields(run.identity),
        "runtime_request_identity": identity_fields(runtime_request.identity),
        "attempt_number": run.attempt_number,
        "fixture_runtime": True,
    }
    if not legacy_ownership_evidence:
        payload.update(
            {
                "run_ownership_identity": identity_fields(claim),
                "artifact_id": launch.artifact_id,
            }
        )
    service._append(launch.run_id, "launched", "run_launched", payload)


def test_abandoned_sweep_checks_live_lease_before_record_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-live-owner-snapshot",
        request_id="request-live-owner-snapshot",
        artifact_id="artifact-live-owner-snapshot",
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(launch)
    root = (tmp_path / ".temper-fixture-output").resolve()

    with claim_run_ownership(root, launch.run_id, claim):
        _persist_synthetic_running_attempt(service, launch, claim)
        restarted = RunService(tmp_path)
        monkeypatch.setattr(restarted.store, "iter_records", lambda: iter(()))

        assert restarted.reconcile_abandoned_runs() == ()
        assert service.status(launch.run_id) is RunLifecycleStatus.RUNNING


def test_abandoned_sweep_leaves_legacy_running_claim_unresolved(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-legacy-running-ownership",
        request_id="request-legacy-running-ownership",
        artifact_id="artifact-legacy-running-ownership",
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(launch)
    root = (tmp_path / ".temper-fixture-output").resolve()
    with claim_run_ownership(root, launch.run_id, claim):
        _persist_synthetic_running_attempt(
            service,
            launch,
            claim,
            legacy_ownership_evidence=True,
        )

    assert RunService(tmp_path).reconcile_abandoned_runs() == ()
    with pytest.raises(RunOwnershipError, match="^run_ownership_unresolved$"):
        released_run_claim_identity(root, launch.run_id)
    assert service.status(launch.run_id) is RunLifecycleStatus.RUNNING


def test_abandoned_sweep_recovers_run_named_like_legacy_control_directory(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="replay",
        request_id="request-control-name-replay",
        artifact_id="artifact-control-name-replay",
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(launch)
    root = (tmp_path / ".temper-fixture-output").resolve()
    with claim_run_ownership(root, launch.run_id, claim):
        _persist_synthetic_running_attempt(service, launch, claim)

    assert RunService(tmp_path).reconcile_abandoned_runs() == (launch.run_id,)
    assert service.status(launch.run_id) is RunLifecycleStatus.INTERRUPTED
    assert released_run_claim_identity(root, launch.run_id) == claim


def test_prelaunch_record_conflict_leaves_run_id_retryable(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-prelaunch-conflict",
        request_id="request-prelaunch-conflict",
        artifact_id="artifact-prelaunch-conflict",
    )
    conflicting_profile = replace(
        foundation.profile,
        library_versions={
            **foundation.profile.library_versions,
            "synthetic_conflict_marker": "v1",
        },
    )
    service = RunService(tmp_path)

    with pytest.raises(ApplicationServiceError, match="run_record_conflict"):
        service.launch(replace(launch, hardware_capability_profile=conflicting_profile))

    assert service._events(launch.run_id) == ()
    with pytest.raises(ApplicationServiceError, match="run_not_found"):
        service.status(launch.run_id)
    assert not any(
        isinstance(stored.record, Run) and stored.record.run_id == launch.run_id
        for stored in service.store.iter_records()
    )
    assert not any(
        isinstance(stored.record, ResolvedRuntimeRequest)
        and stored.record.request_id == launch.request_id
        for stored in service.store.iter_records()
    )

    corrected = service.launch(launch)

    assert corrected.status is RunLifecycleStatus.COMPLETED
    assert service.status(launch.run_id) is RunLifecycleStatus.COMPLETED


def test_resolved_claim_reuse_is_recovered_after_later_process_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-resolved-claim-reused",
        request_id="request-resolved-claim-reused",
        artifact_id="artifact-resolved-claim-reused",
    )
    conflicting_profile = replace(
        foundation.profile,
        library_versions={
            **foundation.profile.library_versions,
            "synthetic_conflict_marker": "v2",
        },
    )
    service = RunService(tmp_path)

    with pytest.raises(ApplicationServiceError, match="run_record_conflict"):
        service.launch(replace(launch, hardware_capability_profile=conflicting_profile))

    root = (tmp_path / ".temper-fixture-output").resolve()
    claim = service.planned_first_attempt_ownership(launch)
    assert released_run_claim_identity(root, launch.run_id) == claim
    original_write = runs_module.write_record_idempotently

    def write_run_then_lose_process(*args: object, **kwargs: object) -> None:
        original_write(*args, **kwargs)  # type: ignore[arg-type]
        record = args[1] if len(args) > 1 else kwargs.get("record")
        if isinstance(record, Run) and record.run_id == launch.run_id:
            raise SystemExit("synthetic loss after resolved-claim Run write")

    with monkeypatch.context() as process_loss:
        process_loss.setattr(
            runs_module, "write_record_idempotently", write_run_then_lose_process
        )
        with pytest.raises(SystemExit, match="resolved-claim Run write"):
            service.launch(launch)

    assert service._events(launch.run_id) == ()
    restarted = RunService(tmp_path)
    assert restarted.reconcile_abandoned_runs() == (launch.run_id,)
    assert restarted.status(launch.run_id) is RunLifecycleStatus.FAILED
    assert released_run_claim_identity(root, launch.run_id) == claim


def test_legacy_v1_prelaunch_prefix_remains_conservatively_unresolved(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-legacy-v1-prelaunch",
        request_id="request-legacy-v1-prelaunch",
        artifact_id="artifact-legacy-v1-prelaunch",
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(launch)
    preflight_result = runs_module.preflight(
        launch.recipe_resolution,
        launch.hardware_requirements,
        launch.execution_target,
        launch.hardware_capability_profile,
        launch.estimate,
    )
    runtime_request, run = service.planned_first_attempt(launch, preflight_result)
    root = (tmp_path / ".temper-fixture-output").resolve()

    with claim_run_ownership(root, launch.run_id, claim):
        service._persist_launch_records(
            launch.hardware_capability_profile,
            runtime_request,
            run,
        )
        service._append(
            launch.run_id,
            "preflight",
            "run_preflight_succeeded",
            {
                "ready": True,
                "preflight_identity": identity_fields(
                    runtime_request.preflight_identity
                ),
                "blocking_reasons": [],
            },
        )

    restarted = RunService(tmp_path)
    assert restarted.reconcile_abandoned_runs() == ()
    assert tuple(event.event_type for event in restarted._events(launch.run_id)) == (
        "run_preflight_succeeded",
    )
    with pytest.raises(ApplicationServiceError, match="^run_lifecycle_incomplete$"):
        restarted.status(launch.run_id)
    with pytest.raises(RunOwnershipError, match="^run_ownership_unresolved$"):
        released_run_claim_identity(root, launch.run_id)


def test_legacy_v1_preflight_blocked_remains_conservatively_unresolved(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-legacy-v1-preflight-blocked",
        request_id="request-legacy-v1-preflight-blocked",
        artifact_id="artifact-legacy-v1-preflight-blocked",
    )
    blocked = replace(
        launch, estimate=replace(launch.estimate, accelerator_memory_bytes=1)
    )
    service = RunService(tmp_path)
    claim = service.planned_first_attempt_ownership(blocked)
    preflight_result = runs_module.preflight(
        blocked.recipe_resolution,
        blocked.hardware_requirements,
        blocked.execution_target,
        blocked.hardware_capability_profile,
        blocked.estimate,
    )
    preflight_identity = content_identity(
        runs_module.PREFLIGHT_EVIDENCE_PROJECTION,
        preflight_result.to_view(),
    )
    root = (tmp_path / ".temper-fixture-output").resolve()

    with claim_run_ownership(root, blocked.run_id, claim):
        service._append(
            blocked.run_id,
            "preflight",
            "run_preflight_blocked",
            {
                "ready": False,
                "preflight_identity": identity_fields(preflight_identity),
                "blocking_reasons": list(preflight_result.blocking_reasons),
            },
        )

    restarted = RunService(tmp_path)
    assert restarted.reconcile_abandoned_runs() == ()
    assert restarted.status(blocked.run_id) is RunLifecycleStatus.PREFLIGHT_BLOCKED
    with pytest.raises(RunOwnershipError, match="^run_ownership_unresolved$"):
        released_run_claim_identity(root, blocked.run_id)


class _FailingAdapter(FixtureAdapter):
    def execute(self, request, *, control=None):
        del request, control
        raise FixtureAdapterError("fixture_failure_injected")


def test_runtime_failure_appends_public_safe_terminal_evidence(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path, adapter=_FailingAdapter())
    launch = foundation.launch(
        run_id="run-failed",
        request_id="request-failed",
        artifact_id="artifact-failed",
    )

    with pytest.raises(ApplicationServiceError, match="fixture_failure_injected"):
        service.launch(launch)

    assert service.status(launch.run_id) is RunLifecycleStatus.FAILED
    event = service._events(launch.run_id)[-1]
    assert event.event_type == "run_failed"
    assert event.payload["failure_code"] == "fixture_failure_injected"
    assert str(tmp_path) not in str(event.payload)


def test_checkpoint_write_failure_terminalizes_the_launched_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)
    launch = foundation.launch(
        run_id="run-checkpoint-write-failed",
        request_id="request-checkpoint-write-failed",
        artifact_id="artifact-checkpoint-write-failed",
    )

    def fail_checkpoint_write(path, data):
        del path, data
        raise SafeIoError("synthetic checkpoint persistence failure")

    monkeypatch.setattr(service, "_write_idempotent", fail_checkpoint_write)

    with pytest.raises(ApplicationServiceError, match="run_output_persistence_failed"):
        service.launch(launch)

    assert service.status(launch.run_id) is RunLifecycleStatus.FAILED
    event = service._events(launch.run_id)[-1]
    assert event.event_type == "run_failed"
    assert event.payload["phase"] == "runtime_output"
    assert event.payload["failure_code"] == "run_output_persistence_failed"


def _abandon_fixture_after_availability(
    service: RunService,
    launch: RunLaunchRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> ContentIdentity:
    claim = service.planned_first_attempt_ownership(launch)
    original_write = runs_module.write_record_idempotently

    def write_then_lose_process(*args: object, **kwargs: object) -> None:
        original_write(*args, **kwargs)  # type: ignore[arg-type]
        record = args[1] if len(args) > 1 else kwargs.get("record")
        if isinstance(record, ArtifactAvailability):
            raise SystemExit("synthetic process loss after artifact availability")

    with monkeypatch.context() as process_loss:
        process_loss.setattr(
            runs_module, "write_record_idempotently", write_then_lose_process
        )
        with pytest.raises(SystemExit, match="after artifact availability"):
            service.launch(launch)
    assert service.status(launch.run_id) is RunLifecycleStatus.RUNNING
    return claim


@pytest.mark.parametrize("member_state", ["intact", "missing"])
def test_abandoned_artifact_honors_durable_ingestion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_state: str,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id=f"run-durable-ingestion-failure-{member_state}",
        request_id=f"request-durable-ingestion-failure-{member_state}",
        artifact_id=f"artifact-durable-ingestion-failure-{member_state}",
    )
    service = RunService(tmp_path)
    claim = _abandon_fixture_after_availability(service, launch, monkeypatch)
    if member_state == "missing":
        (
            tmp_path
            / ".temper-fixture-output"
            / "artifacts"
            / launch.artifact_id
            / "adapter.bin"
        ).unlink()
    service._append(
        launch.run_id,
        "artifact-ingestion-failed",
        "artifact_ingestion_failed",
        {
            "failure_code": "run_abandoned_artifact_unrecoverable",
            "verified_artifact": False,
        },
    )

    restarted = RunService(tmp_path)
    assert restarted.reconcile_abandoned_runs() == (launch.run_id,)
    assert restarted.status(launch.run_id) is RunLifecycleStatus.FAILED
    assert (
        released_run_claim_identity(
            (tmp_path / ".temper-fixture-output").resolve(), launch.run_id
        )
        == claim
    )
    events = restarted._events(launch.run_id)
    assert sum(event.event_type == "artifact_ingestion_failed" for event in events) == 1
    assert sum(event.event_type == "run_failed" for event in events) == 1
    assert not any(event.event_type == "run_completed" for event in events)
    artifact = next(
        stored.record
        for stored in restarted.store.iter_records()
        if isinstance(stored.record, Artifact)
        and stored.record.artifact_id == launch.artifact_id
    )
    availabilities = tuple(
        stored.record
        for stored in restarted.store.iter_records()
        if isinstance(stored.record, ArtifactAvailability)
        and stored.record.artifact == record_reference(artifact)
    )
    superseded = {
        availability.supersedes.identity
        for availability in availabilities
        if availability.supersedes is not None
    }
    current = tuple(
        availability
        for availability in availabilities
        if availability.identity not in superseded
    )
    assert len(current) == 1
    assert current[0].state is AvailabilityState.UNAVAILABLE


def test_abandoned_ingestion_failure_survives_second_process_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-ingestion-failure-second-loss",
        request_id="request-ingestion-failure-second-loss",
        artifact_id="artifact-ingestion-failure-second-loss",
    )
    service = RunService(tmp_path)
    claim = _abandon_fixture_after_availability(service, launch, monkeypatch)
    (
        tmp_path
        / ".temper-fixture-output"
        / "artifacts"
        / launch.artifact_id
        / "adapter.bin"
    ).unlink()
    original_append = RunService._append

    def append_then_lose_process(
        run_service: RunService,
        run_id: str,
        key: str,
        event_type: str,
        payload: object,
    ) -> object:
        event = original_append(
            run_service,
            run_id,
            key,
            event_type,
            payload,  # type: ignore[arg-type]
        )
        if event_type == "artifact_ingestion_failed":
            raise SystemExit("synthetic process loss after ingestion failure")
        return event

    with monkeypatch.context() as second_loss:
        second_loss.setattr(RunService, "_append", append_then_lose_process)
        with pytest.raises(SystemExit, match="after ingestion failure"):
            RunService(tmp_path).reconcile_abandoned_runs()

    after_loss = RunService(tmp_path)
    assert after_loss.status(launch.run_id) is RunLifecycleStatus.RUNNING
    assert (
        sum(
            event.event_type == "artifact_ingestion_failed"
            for event in after_loss._events(launch.run_id)
        )
        == 1
    )
    assert after_loss.reconcile_abandoned_runs() == (launch.run_id,)
    assert after_loss.status(launch.run_id) is RunLifecycleStatus.FAILED
    assert (
        released_run_claim_identity(
            (tmp_path / ".temper-fixture-output").resolve(), launch.run_id
        )
        == claim
    )
    events = after_loss._events(launch.run_id)
    assert sum(event.event_type == "artifact_ingestion_failed" for event in events) == 1
    assert sum(event.event_type == "run_failed" for event in events) == 1
    assert not any(event.event_type == "run_completed" for event in events)


@pytest.mark.parametrize(
    ("failed_event", "expected_phase"),
    [
        ("run_progress", "runtime_output"),
        ("artifact_ingestion_verified", "artifact_ingestion"),
        ("run_completed", "completion"),
    ],
)
def test_post_launch_event_append_failure_terminalizes_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_event: str,
    expected_phase: str,
) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)
    launch = foundation.launch(
        run_id=f"run-{failed_event}-append-failed",
        request_id=f"request-{failed_event}-append-failed",
        artifact_id=f"artifact-{failed_event}-append-failed",
    )
    original_append = service._append
    injected = False

    def fail_one_event(run_id, key, event_type, payload):
        nonlocal injected
        if event_type == failed_event and not injected:
            injected = True
            raise EvidenceError("event_append_failed")
        return original_append(run_id, key, event_type, payload)

    monkeypatch.setattr(service, "_append", fail_one_event)

    with pytest.raises(ApplicationServiceError, match="event_append_failed"):
        service.launch(launch)

    assert injected is True
    assert service.status(launch.run_id) is RunLifecycleStatus.FAILED
    event = service._events(launch.run_id)[-1]
    assert event.event_type == "run_failed"
    assert event.payload["phase"] == expected_phase
    assert event.payload["failure_code"] == "event_append_failed"
    if failed_event in {"artifact_ingestion_verified", "run_completed"}:
        artifact = next(
            stored.record
            for stored in service.store.iter_records()
            if isinstance(stored.record, Artifact)
            and stored.record.artifact_id == launch.artifact_id
        )
        availabilities = tuple(
            stored.record
            for stored in service.store.iter_records()
            if isinstance(stored.record, ArtifactAvailability)
            and stored.record.artifact == record_reference(artifact)
        )
        superseded = {
            availability.supersedes.identity
            for availability in availabilities
            if availability.supersedes is not None
        }
        current = tuple(
            availability
            for availability in availabilities
            if availability.identity not in superseded
        )
        assert len(current) == 1
        assert current[0].state is AvailabilityState.UNAVAILABLE


def test_available_artifact_cannot_be_consumed_before_run_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = _foundation(tmp_path)
    launch = foundation.launch(
        run_id="run-available-before-completion",
        request_id="request-available-before-completion",
        artifact_id="artifact-available-before-completion",
    )
    service = RunService(tmp_path)
    original_write = runs_module.write_record_idempotently
    consumption_blocked = False

    def inspect_then_fail(*args: object, **kwargs: object) -> None:
        nonlocal consumption_blocked
        original_write(*args, **kwargs)  # type: ignore[arg-type]
        record = args[1] if len(args) > 1 else kwargs.get("record")
        if (
            isinstance(record, ArtifactAvailability)
            and record.state is AvailabilityState.AVAILABLE
            and record.artifact.logical_id == launch.artifact_id
            and not consumption_blocked
        ):
            artifact = next(
                stored.record
                for stored in TypedEvidenceStore(tmp_path).iter_records()
                if isinstance(stored.record, Artifact)
                and stored.record.artifact_id == launch.artifact_id
            )
            with pytest.raises(
                ApplicationServiceError, match="^local_use_artifact_unavailable$"
            ):
                LocalUseService(tmp_path).inspect_artifact(
                    artifact,
                    foundation.model,
                    foundation.group,
                    foundation.target,
                )
            consumption_blocked = True
            raise EvidenceError("synthetic_availability_boundary_failure")

    with monkeypatch.context() as availability_failure:
        availability_failure.setattr(
            runs_module, "write_record_idempotently", inspect_then_fail
        )
        with pytest.raises(
            ApplicationServiceError, match="^synthetic_availability_boundary_failure$"
        ):
            service.launch(launch)

    assert consumption_blocked is True
    assert service.status(launch.run_id) is RunLifecycleStatus.FAILED
    artifact = next(
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, Artifact)
        and stored.record.artifact_id == launch.artifact_id
    )
    availabilities = tuple(
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, ArtifactAvailability)
        and stored.record.artifact == record_reference(artifact)
    )
    superseded = {
        availability.supersedes.identity
        for availability in availabilities
        if availability.supersedes is not None
    }
    current = tuple(
        availability
        for availability in availabilities
        if availability.identity not in superseded
    )
    assert len(current) == 1
    assert current[0].state is AvailabilityState.UNAVAILABLE


@pytest.mark.parametrize(
    ("ambiguous_event", "adapter_fails", "expected_terminal"),
    [
        ("run_launched", False, "run_completed"),
        ("run_progress", False, "run_completed"),
        ("run_completed", False, "run_completed"),
        ("run_failed", True, "run_failed"),
    ],
)
def test_commit_then_raise_event_append_is_reconciled_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambiguous_event: str,
    adapter_fails: bool,
    expected_terminal: str,
) -> None:
    foundation = _foundation(tmp_path)
    adapter = _FailingAdapter() if adapter_fails else FixtureAdapter()
    service = RunService(tmp_path, adapter=adapter)
    launch = foundation.launch(
        run_id=f"run-{ambiguous_event}-commit-unknown",
        request_id=f"request-{ambiguous_event}-commit-unknown",
        artifact_id=f"artifact-{ambiguous_event}-commit-unknown",
    )
    original_append = service.store.append_event
    injected = 0
    injected_key = None

    def commit_then_raise(stream_id, request):
        nonlocal injected, injected_key
        durable = original_append(stream_id, request)
        if request.event_type == ambiguous_event and injected == 0:
            injected += 1
            injected_key = request.idempotency_key
            raise EvidenceError("synthetic_commit_outcome_unknown")
        return durable

    monkeypatch.setattr(service.store, "append_event", commit_then_raise)

    if adapter_fails:
        with pytest.raises(ApplicationServiceError, match="fixture_failure_injected"):
            service.launch(launch)
        expected_status = RunLifecycleStatus.FAILED
    else:
        result = service.launch(launch)
        expected_status = RunLifecycleStatus.COMPLETED
        assert result.status is expected_status

    events = service._events(launch.run_id)
    terminal_types = {
        "run_preflight_blocked",
        "run_cancelled",
        "run_interrupted",
        "run_completed",
        "run_failed",
    }
    assert injected == 1
    assert injected_key is not None
    assert service.status(launch.run_id) is expected_status
    assert sum(event.idempotency_key == injected_key for event in events) == 1
    assert [
        event.event_type for event in events if event.event_type in terminal_types
    ] == [expected_terminal]
    assert events[-1].event_type == expected_terminal


@pytest.mark.parametrize("record_kind", [Artifact, ArtifactAvailability])
def test_artifact_record_conflicts_terminalize_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind,
) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)
    launch = foundation.launch(
        run_id=f"run-{record_kind.RECORD_TYPE}-conflict",
        request_id=f"request-{record_kind.RECORD_TYPE}-conflict",
        artifact_id=f"artifact-{record_kind.RECORD_TYPE}-conflict",
    )
    original_check = runs_module.require_no_conflicting_logical_revision

    def inject_conflict(store, record, *, conflict_code):
        if isinstance(record, record_kind):
            raise ApplicationServiceError("artifact_record_conflict")
        return original_check(store, record, conflict_code=conflict_code)

    monkeypatch.setattr(
        runs_module, "require_no_conflicting_logical_revision", inject_conflict
    )

    with pytest.raises(ApplicationServiceError, match="artifact_record_conflict"):
        service.launch(launch)

    assert service.status(launch.run_id) is RunLifecycleStatus.FAILED
    events = service._events(launch.run_id)
    assert events[-2].event_type == "artifact_ingestion_failed"
    assert events[-1].event_type == "run_failed"
    assert events[-1].payload["phase"] == "artifact_ingestion"
    assert events[-1].payload["failure_code"] == "artifact_record_conflict"


def test_final_step_interruption_fails_without_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    service = RunService(tmp_path)
    launch = foundation.launch(
        run_id="run-final-step-interruption",
        request_id="request-final-step-interruption",
        artifact_id="artifact-final-step-interruption",
    )

    with pytest.raises(ApplicationServiceError, match="fixture_control_out_of_range"):
        service.launch(
            launch,
            control=FixtureControl(
                interrupt_after_step=foundation.resolution.training_steps
            ),
        )

    assert service.status(launch.run_id) is RunLifecycleStatus.FAILED
    events = service._events(launch.run_id)
    assert not any(event.event_type == "run_checkpoint" for event in events)
    assert events[-1].event_type == "run_failed"
    assert events[-1].payload["failure_code"] == "fixture_control_out_of_range"


def test_slice_five_fails_closed_on_future_quality_modes(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    launch = replace(
        foundation.launch(
            run_id="run-quality-not-implemented",
            request_id="request-quality-not-implemented",
            artifact_id="artifact-quality-not-implemented",
        ),
        evaluation_mode=EvaluationMode.LIGHT_EVALUATION,
    )

    with pytest.raises(
        ApplicationServiceError, match="run_evaluation_mode_not_supported"
    ):
        RunService(tmp_path).launch(launch)


def test_library_runtime_preserves_run_artifact_local_use_and_export_contracts(
    tmp_path: Path,
) -> None:
    foundation, backend, sources, adapter = _library_foundation(tmp_path)
    service = RunService(tmp_path, adapter=adapter)
    launch = foundation.launch(
        run_id="run-library-runtime",
        request_id="request-library-runtime",
        artifact_id="artifact-library-runtime",
    )

    result = service.launch(launch)

    assert result.status is RunLifecycleStatus.COMPLETED
    assert result.artifact is not None
    assert result.integrity is not None
    assert backend.train_calls == 1
    assert (
        service.reconcile_runtime_controller(result.run.run_id).state
        is ControllerState.COMPLETED
    )
    event_types = tuple(
        event.event_type for event in service._events(result.run.run_id)
    )
    assert "run_worker_message" in event_types
    assert "runtime_transfer_verified" in event_types
    assert event_types[-1] == "run_completed"
    config = loads_canonical_json(
        (
            tmp_path
            / ".temper-fixture-output"
            / "artifacts"
            / launch.artifact_id
            / "adapter_config.json"
        ).read_bytes()
    )
    assert config["runtime_kind"] == "library"
    assert config["runtime_identity"] == {
        "algorithm": adapter.runtime_identity.algorithm,
        "value": adapter.runtime_identity.value,
    }

    reopened = service.reopen_completed(launch)
    assert reopened.artifact == result.artifact
    assert backend.train_calls == 1

    inference_runtime = LibraryInferenceRuntime(
        backend, sources, adapter.runtime_identity
    )
    drifted_capability = replace(
        backend.probe(),
        library_versions={
            **backend.probe().library_versions,
            "transformers": "2.test",
        },
    )
    with pytest.raises(LibraryRuntimeError, match="library_inference_runtime_mismatch"):
        LibraryInferenceRuntime(
            DeterministicLibraryBackend(drifted_capability),
            sources,
            adapter.runtime_identity,
        )
    local = LocalUseService(tmp_path, runtime=inference_runtime)
    focused = local.focused(
        LocalUseRequest(
            artifact=result.artifact,
            base_model_revision=foundation.model,
            compatibility_group=foundation.group,
            execution_target=foundation.target,
            settings=InferenceSettings(),
            inputs=({"prompt": "Synthetic focused input"},),
        )
    )
    batch = local.batch(
        LocalUseRequest(
            artifact=result.artifact,
            base_model_revision=foundation.model,
            compatibility_group=foundation.group,
            execution_target=foundation.target,
            settings=InferenceSettings(),
            inputs=({"prompt": "Synthetic single-item batch input"},),
            session_id="library-local-session",
        )
    )
    assert focused.ephemeral is True
    assert batch.session is not None
    adapter_bytes = (
        tmp_path
        / ".temper-fixture-output"
        / "artifacts"
        / launch.artifact_id
        / "adapter.bin"
    ).read_bytes()
    evaluated = inference_runtime.infer_verified(
        FixtureInferenceRequest(
            adapter_bytes,
            result.artifact.content_identity,
            InferenceSettings(),
            ({"prompt": "Synthetic evaluation input"},),
        ),
        resolution=foundation.resolution,
        adapter_config=config,
        operation=RuntimeOperation.EVALUATE,
    )
    assert len(evaluated.outputs) == 1
    mismatched_config = dict(config)
    mismatched_config["tokenizer_identity"] = {
        "algorithm": "sha256",
        "value": "e" * 64,
    }
    with pytest.raises(LibraryRuntimeError, match="library_inference_source_mismatch"):
        inference_runtime.infer_verified(
            FixtureInferenceRequest(
                adapter_bytes,
                result.artifact.content_identity,
                InferenceSettings(),
                ({"prompt": "Synthetic mismatched source input"},),
            ),
            resolution=foundation.resolution,
            adapter_config=mismatched_config,
            operation=RuntimeOperation.EVALUATE,
        )

    class PrivateDiagnosticBackend(DeterministicLibraryBackend):
        def infer(self, **kwargs):
            del kwargs
            raise OSError("<private-model-source>/tokenizer.json is malformed")

    private_backend = PrivateDiagnosticBackend(backend.probe())
    private_runtime = LibraryInferenceRuntime(
        private_backend,
        sources,
        adapter.runtime_identity,
    )
    with pytest.raises(LibraryRuntimeError) as failure:
        private_runtime.infer_verified(
            FixtureInferenceRequest(
                adapter_bytes,
                result.artifact.content_identity,
                InferenceSettings(),
                ({"prompt": "Synthetic private diagnostic boundary input"},),
            ),
            resolution=foundation.resolution,
            adapter_config=config,
            operation=RuntimeOperation.EVALUATE,
        )
    assert failure.value.code == "library_inference_failed"
    assert str(failure.value) == "library_inference_failed"
    assert backend.inference_calls == 3
    assert backend.inference_operations == [
        RuntimeOperation.INFER_FOCUSED,
        RuntimeOperation.INFER_BATCH,
        RuntimeOperation.EVALUATE,
    ]

    exported = local.export(
        AdapterExportRequest(
            "library-export",
            result.artifact,
            foundation.model,
            foundation.group,
            foundation.target,
        )
    )
    assert exported.record.export_format == "temper_library_adapter_bundle"
    assert (
        local.verify_export(
            exported.record,
            artifact=result.artifact,
            base_model_revision=foundation.model,
            compatibility_group=foundation.group,
            execution_target=foundation.target,
        )
        == exported.integrity
    )


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    (
        pytest.param(None, RunLifecycleStatus.COMPLETED, id="intact"),
        pytest.param("missing", RunLifecycleStatus.FAILED, id="missing-member"),
        pytest.param("mutated", RunLifecycleStatus.FAILED, id="mutated-member"),
    ),
)
def test_abandoned_library_completion_reverifies_current_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None,
    expected_status: RunLifecycleStatus,
) -> None:
    foundation, _, _, adapter = _library_foundation(tmp_path)
    launch = foundation.launch(
        run_id=f"run-library-abandoned-{mutation or 'intact'}",
        request_id=f"request-library-abandoned-{mutation or 'intact'}",
        artifact_id=f"artifact-library-abandoned-{mutation or 'intact'}",
    )
    service = RunService(tmp_path, adapter=adapter)
    claim = service.planned_first_attempt_ownership(launch)
    original_append = RunService._append

    def append_then_lose_process(
        run_service: RunService,
        run_id: str,
        key: str,
        event_type: str,
        payload: object,
    ) -> object:
        event = original_append(
            run_service,
            run_id,
            key,
            event_type,
            payload,  # type: ignore[arg-type]
        )
        if run_id == launch.run_id and event_type == "artifact_ingestion_verified":
            raise SystemExit("synthetic library process loss after ingestion")
        return event

    with monkeypatch.context() as process_loss:
        process_loss.setattr(RunService, "_append", append_then_lose_process)
        with pytest.raises(SystemExit, match="library process loss"):
            service.launch(launch)

    assert service.status(launch.run_id) is RunLifecycleStatus.RUNNING
    member = (
        tmp_path
        / ".temper-fixture-output"
        / "artifacts"
        / launch.artifact_id
        / "adapter.bin"
    )
    if mutation == "missing":
        member.unlink()
    elif mutation == "mutated":
        member.write_bytes(b"synthetic mutated adapter bytes")

    restarted = RunService(tmp_path)
    assert restarted.reconcile_abandoned_runs() == (launch.run_id,)
    assert restarted.status(launch.run_id) is expected_status
    assert (
        released_run_claim_identity(
            (tmp_path / ".temper-fixture-output").resolve(), launch.run_id
        )
        == claim
    )
    events = restarted._events(launch.run_id)
    terminal = events[-1]
    artifacts = tuple(
        stored.record
        for stored in TypedEvidenceStore(tmp_path).iter_records()
        if isinstance(stored.record, Artifact)
        and stored.record.artifact_id == launch.artifact_id
    )
    assert len(artifacts) == 1
    availabilities = tuple(
        stored.record
        for stored in TypedEvidenceStore(tmp_path).iter_records()
        if isinstance(stored.record, ArtifactAvailability)
        and stored.record.artifact == record_reference(artifacts[0])
    )
    superseded = {
        availability.supersedes.identity
        for availability in availabilities
        if availability.supersedes is not None
    }
    current = tuple(
        availability
        for availability in availabilities
        if availability.identity not in superseded
    )
    assert len(current) == 1
    if expected_status is RunLifecycleStatus.COMPLETED:
        assert terminal.event_type == "run_completed"
        assert current[0].state is AvailabilityState.AVAILABLE
    else:
        assert terminal.event_type == "run_failed"
        assert (
            terminal.payload.get("failure_code")
            == "run_abandoned_artifact_unrecoverable"
        )
        assert current[0].state is AvailabilityState.UNAVAILABLE


def test_library_runtime_cancellation_interruption_and_recovery_release_resources(
    tmp_path: Path,
) -> None:
    foundation, backend, _, adapter = _library_foundation(tmp_path)
    service = RunService(tmp_path, adapter=adapter)
    cancelled = service.launch(
        foundation.launch(
            run_id="run-library-cancelled",
            request_id="request-library-cancelled",
            artifact_id="artifact-library-cancelled",
        ),
        control=FixtureControl(cancel_after_step=2),
    )
    assert cancelled.status is RunLifecycleStatus.CANCELLED
    assert not adapter.resources.leases

    interrupted = service.launch(
        foundation.launch(
            run_id="run-library-interrupted",
            request_id="request-library-interrupted",
            artifact_id="artifact-library-interrupted",
        ),
        control=FixtureControl(interrupt_after_step=3),
    )
    assert interrupted.status is RunLifecycleStatus.INTERRUPTED
    assert interrupted.checkpoints
    assert (
        service.reconcile_runtime_controller(interrupted.run.run_id).state
        is ControllerState.INTERRUPTED
    )
    assert not adapter.resources.leases

    recovered = service.recover(
        RunRecoveryRequest(
            foundation.launch(
                run_id="run-library-recovered",
                request_id="request-library-recovered",
                artifact_id="artifact-library-recovered",
            ),
            interrupted.run,
            interrupted.checkpoints[-1].checkpoint_identity,
        )
    )
    assert recovered.status is RunLifecycleStatus.COMPLETED
    assert recovered.runtime_request.starting_step == interrupted.checkpoints[-1].step
    assert recovered.run.attempt_number == 2
    assert backend.train_calls == 3
    assert not adapter.resources.leases


def test_terminal_library_checkpoint_is_retained_but_cannot_resume(
    tmp_path: Path,
) -> None:
    foundation, base_backend, sources, _ = _library_foundation(tmp_path)

    class FinalCheckpointInterruptedBackend(DeterministicLibraryBackend):
        def train(self, **kwargs):
            completed = super().train(**kwargs)
            return LibraryTrainingResult(
                None,
                None,
                None,
                completed.progress,
                completed.checkpoints,
                interrupted=True,
            )

    backend = FinalCheckpointInterruptedBackend(base_backend.probe())
    service = RunService(
        tmp_path,
        adapter=LibraryAdapter(backend, sources, capability=backend.probe()),
    )
    interrupted = service.launch(
        foundation.launch(
            run_id="run-library-final-checkpoint",
            request_id="request-library-final-checkpoint",
            artifact_id="artifact-library-final-checkpoint",
        )
    )

    assert interrupted.status is RunLifecycleStatus.INTERRUPTED
    final_checkpoint = interrupted.checkpoints[-1]
    assert final_checkpoint.step == foundation.resolution.training_steps
    assert final_checkpoint.resume_compatible is False
    final_event = service._checkpoint_event(
        interrupted.run.run_id, final_checkpoint.checkpoint_identity
    )
    assert final_event.payload["resume_compatible"] is False
    with pytest.raises(
        ApplicationServiceError, match="run_recovery_checkpoint_incompatible"
    ):
        service.recover(
            RunRecoveryRequest(
                foundation.launch(
                    run_id="run-library-final-checkpoint-recovery",
                    request_id="request-library-final-checkpoint-recovery",
                    artifact_id="artifact-library-final-checkpoint-recovery",
                ),
                interrupted.run,
                final_checkpoint.checkpoint_identity,
            )
        )
    assert backend.train_calls == 1


def test_real_backend_rejects_decoded_terminal_checkpoint_step(
    tmp_path: Path,
) -> None:
    foundation, _, _, _ = _library_foundation(tmp_path)
    rng_state = object()
    serialized_state = {
        "schema_version": "v2",
        "step": foundation.resolution.training_steps,
        "batches_consumed": foundation.resolution.training_steps,
        "recipe_resolution": record_reference(foundation.resolution).to_dict(),
        "adapter_state": {},
        "optimizer_state": {},
        "scheduler_state": {},
        "torch_rng_state": rng_state,
        "cuda_rng_states": (),
        "accelerator_scaler_state": None,
    }

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            return serialized_state

        @staticmethod
        def is_tensor(value):
            return value is rng_state

    with pytest.raises(LibraryRuntimeError, match="library_checkpoint_restore_failed"):
        library_backend_module._checkpoint_state(
            FakeTorch(), b"synthetic-checkpoint", foundation.resolution
        )


@pytest.mark.parametrize("failure", [EOFError(), pickle.UnpicklingError("synthetic")])
def test_real_backend_normalizes_checkpoint_deserialization_failures(
    tmp_path: Path, failure: Exception
) -> None:
    foundation, _, _, _ = _library_foundation(tmp_path)

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            raise failure

    with pytest.raises(LibraryRuntimeError, match="library_checkpoint_restore_failed"):
        library_backend_module._checkpoint_state(
            FakeTorch(), b"synthetic-malformed-checkpoint", foundation.resolution
        )


def test_checkpoint_loader_position_matches_accumulation_and_loader_end(
    tmp_path: Path,
) -> None:
    foundation, _, _, _ = _library_foundation(tmp_path)
    resolution = replace(foundation.resolution, gradient_accumulation=2)
    loader = ("batch-a", "batch-b", "batch-c")

    library_backend_module._validate_checkpoint_loader_position(
        {"step": 2, "batches_consumed": 3}, loader, resolution
    )
    with pytest.raises(LibraryRuntimeError, match="library_checkpoint_restore_failed"):
        library_backend_module._validate_checkpoint_loader_position(
            {"step": 1, "batches_consumed": 1}, loader, resolution
        )


def test_library_runtime_has_one_live_owner_per_run(tmp_path: Path) -> None:
    foundation, base_backend, sources, _ = _library_foundation(tmp_path)

    class BlockingBackend(DeterministicLibraryBackend):
        def __init__(self, capability: LibraryCapability) -> None:
            super().__init__(capability)
            self.started = Event()
            self.release = Event()

        def train(self, **kwargs):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test backend release timed out")
            return super().train(**kwargs)

    backend = BlockingBackend(base_backend.probe())
    adapter = LibraryAdapter(backend, sources, capability=backend.probe())
    service = RunService(tmp_path, adapter=adapter)
    launch = foundation.launch(
        run_id="run-library-single-owner",
        request_id="request-library-single-owner",
        artifact_id="artifact-library-single-owner",
    )
    runtime_request, run = service._build_execution_records(
        launch,
        ContentIdentity("sha256", "9" * 64),
        attempt_number=1,
        retry_of=None,
        recovery_checkpoint=None,
    )
    request = FixtureAdapterRequest(
        foundation.experiment,
        foundation.resolution,
        foundation.prepared.version,
        foundation.prepared.rendered_bytes,
        runtime_request,
        run,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.execute, request)
        assert backend.started.wait(timeout=5)
        try:
            with pytest.raises(LibraryRuntimeError, match="library_run_already_active"):
                adapter.execute(request)
        finally:
            backend.release.set()
        assert future.result(timeout=10).completed is True

    assert backend.train_calls == 1
    assert not adapter.resources.leases


def test_run_service_serializes_one_run_across_adapter_instances(
    tmp_path: Path,
) -> None:
    foundation, base_backend, sources, _ = _library_foundation(tmp_path)

    class BlockingBackend(DeterministicLibraryBackend):
        def __init__(self, capability: LibraryCapability) -> None:
            super().__init__(capability)
            self.started = Event()
            self.release = Event()

        def train(self, **kwargs):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test backend release timed out")
            return super().train(**kwargs)

    backend = BlockingBackend(base_backend.probe())
    first_service = RunService(
        tmp_path,
        adapter=LibraryAdapter(backend, sources, capability=backend.probe()),
    )
    second_service = RunService(
        tmp_path,
        adapter=LibraryAdapter(backend, sources, capability=backend.probe()),
    )
    launch = foundation.launch(
        run_id="run-library-cross-service-owner",
        request_id="request-library-cross-service-owner",
        artifact_id="artifact-library-cross-service-owner",
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first_service.launch, launch)
        assert backend.started.wait(timeout=5)
        try:
            with pytest.raises(
                ApplicationServiceError, match="run_ownership_unavailable"
            ):
                second_service.launch(launch)
        finally:
            backend.release.set()
        assert future.result(timeout=10).status is RunLifecycleStatus.COMPLETED

    with pytest.raises(ApplicationServiceError, match="run_id_already_used"):
        second_service.launch(launch)
    assert backend.train_calls == 1


def test_local_use_distinguishes_ephemeral_and_saved_sessions_and_batches(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    artifact = _one(TypedEvidenceStore(tmp_path), Artifact)
    service = LocalUseService(tmp_path)
    settings = InferenceSettings(0, 32, 9)
    before = sum(
        isinstance(stored.record, LocalUseSession)
        for stored in service.store.iter_records()
    )
    ephemeral_request = LocalUseRequest(
        artifact,
        foundation.model,
        foundation.group,
        foundation.target,
        settings,
        ({"text": "Synthetic ephemeral prompt"},),
    )

    first = service.focused(ephemeral_request)
    second = service.focused(ephemeral_request)
    saved = service.focused(replace(ephemeral_request, session_id="session-unit-saved"))
    batch = service.batch(
        replace(
            ephemeral_request,
            inputs=(
                {"text": "Synthetic batch one"},
                {"text": "Synthetic batch two"},
            ),
        )
    )

    assert first.ephemeral is True
    assert first.inference == second.inference
    assert saved.ephemeral is False
    assert saved.session is not None
    assert len(batch.inference.outputs) == 2
    after = sum(
        isinstance(stored.record, LocalUseSession)
        for stored in service.store.iter_records()
    )
    assert after == before + 1


def test_local_use_rejects_artifact_outside_selected_compatibility_group(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    store = TypedEvidenceStore(tmp_path)
    artifact = _one(store, Artifact)
    incompatible = replace(foundation.group, group_id="group-incompatible")
    store.write_record(incompatible)

    with pytest.raises(
        ApplicationServiceError, match="local_use_compatibility_group_mismatch"
    ):
        LocalUseService(tmp_path).focused(
            LocalUseRequest(
                artifact,
                foundation.model,
                incompatible,
                foundation.target,
                InferenceSettings(),
                ({"text": "Synthetic incompatible prompt"},),
            )
        )


def test_corrupt_export_is_rejected_and_never_claims_deployment(tmp_path: Path) -> None:
    foundation = _foundation(tmp_path)
    artifact = _one(TypedEvidenceStore(tmp_path), Artifact)
    service = LocalUseService(tmp_path)
    exported = service.export(
        AdapterExportRequest(
            "export-corrupt-unit",
            artifact,
            foundation.model,
            foundation.group,
            foundation.target,
        )
    )
    view = exported.to_view()
    assert view["hosted_deployment"] is False
    assert view["deployment_ready"] is False
    (exported.local_root / "integrity-manifest.json").write_bytes(b"corrupt")

    with pytest.raises(
        ApplicationServiceError, match="export_manifest_identity_mismatch"
    ):
        service.verify_export(
            exported.record,
            artifact=artifact,
            base_model_revision=foundation.model,
            compatibility_group=foundation.group,
            execution_target=foundation.target,
        )
