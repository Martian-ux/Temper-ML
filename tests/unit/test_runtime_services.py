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
from temper_ml.cli import _FixtureTokenizer, _fixture_workflow
from temper_ml.domain.artifacts import Artifact, ArtifactAvailability
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
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import record_reference
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


@pytest.mark.parametrize(
    ("failed_event", "expected_phase"),
    [("run_progress", "runtime_output"), ("run_completed", "completion")],
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
