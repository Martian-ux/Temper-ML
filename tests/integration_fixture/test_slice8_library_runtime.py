from dataclasses import replace

from temper_ml.app_services.experiments import (
    ExperimentFreezeRequest,
    ExperimentService,
)
from temper_ml.app_services.fixture_journey import FixtureJourneyService
from temper_ml.app_services.local_use import LocalUseRequest, LocalUseService
from temper_ml.app_services.runs import (
    RunLaunchRequest,
    RunLifecycleStatus,
    RunService,
)
from temper_ml.domain.records import record_reference
from temper_ml.domain.runs import EvaluationMode
from temper_ml.runtime.fixture_inference import InferenceSettings
from temper_ml.runtime.library_adapter import (
    LibraryAdapter,
    LibraryInferenceRuntime,
    LibraryRuntimeSources,
)
from temper_ml.runtime.library_backend import LibraryCapability
from temper_ml.runtime.library_double import DeterministicLibraryBackend
from temper_ml.store.evidence import TypedEvidenceStore


def test_slice8_library_double_uses_existing_run_artifact_and_local_use_contracts(
    tmp_path,
) -> None:
    journey = FixtureJourneyService(tmp_path)
    journey.setup_project()
    journey.import_dataset()
    journey.resolve_candidates()
    state = journey.state
    assert state.opened is not None
    assert state.model is not None
    assert state.prepared is not None
    assert state.requirements is not None
    assert state.target is not None
    assert state.group is not None
    candidate = state.candidates[0]
    original_experiment = ExperimentService(tmp_path).freeze(
        ExperimentFreezeRequest(
            experiment_id=candidate.experiment_id,
            opened_project=state.opened,
            dataset_version=state.prepared.version.identity,
            base_model_revision=state.model,
            recipe=candidate.recipe,
            recipe_resolution=candidate.resolution,
            compatibility_group=state.group,
            hardware_requirements=state.requirements,
            execution_target=state.target,
        )
    )
    versions = {
        "accelerate": "1.test",
        "peft": "1.test",
        "torch": "1.test",
        "transformers": "1.test",
    }
    capability = LibraryCapability(
        accelerator_backend=state.target.accelerator_backend,
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
        (tmp_path / "private-model-cache").resolve(),
        (tmp_path / "private-tokenizer-cache").resolve(),
        (tmp_path / ".temper-fixture-output" / "library-staging").resolve(),
        state.target.target_class,
        record_reference(state.model),
        state.model.tokenizer_identity,
    )
    adapter = LibraryAdapter(backend, sources, capability=capability)
    resolution = replace(
        candidate.resolution,
        resolution_id="resolution-slice8-integration",
        library_versions=versions,
    )
    experiment = replace(
        original_experiment,
        experiment_id="experiment-slice8-integration",
        recipe_resolution=record_reference(resolution),
    )
    profile = adapter.capability_profile("profile-slice8-integration", state.target)
    store = TypedEvidenceStore(tmp_path)
    for record in (resolution, experiment, profile):
        store.write_record(record)
    launch = RunLaunchRequest(
        run_id="run-slice8-integration",
        request_id="request-slice8-integration",
        artifact_id="artifact-slice8-integration",
        experiment=experiment,
        recipe_resolution=resolution,
        prepared_dataset=state.prepared,
        base_model_revision=state.model,
        compatibility_group=state.group,
        hardware_requirements=state.requirements,
        execution_target=state.target,
        hardware_capability_profile=profile,
        estimate=candidate.estimate,
        evaluation_mode=EvaluationMode.NO_QUALITY_EVALUATION,
    )
    runs = RunService(tmp_path, adapter=adapter)

    completed = runs.launch(launch)
    reopened = runs.reopen_completed(launch)

    assert completed.status is RunLifecycleStatus.COMPLETED
    assert completed.artifact is not None
    assert reopened == completed
    assert backend.train_calls == 1

    local_use = LocalUseService(
        tmp_path,
        runtime=LibraryInferenceRuntime(backend, sources, adapter.runtime_identity),
    )
    focused = local_use.focused(
        LocalUseRequest(
            artifact=completed.artifact,
            base_model_revision=state.model,
            compatibility_group=state.group,
            execution_target=state.target,
            settings=InferenceSettings(),
            inputs=({"prompt": "Synthetic Slice 8 integration input"},),
        )
    )

    assert focused.ephemeral is True
    assert len(focused.inference.outputs) == 1
    assert backend.inference_calls == 1
