from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from temper_ml.app_services.datasets import DatasetImportRequest, DatasetService
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import (
    CompatibilityGroup,
    RuntimeTargetConstraint,
)
from temper_ml.domain.datasets import (
    DeduplicationRule,
    FieldMapping,
    FilterRule,
    RendererSpec,
    SplitPart,
    SplitRule,
)
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import RecordReference, record_reference
from temper_ml.domain.runs import EvaluationMode, ResolvedRuntimeRequest, Run
from temper_ml.runtime.fixture_adapter import (
    FIXTURE_ARTIFACT_MEMBERS,
    FIXTURE_RUNTIME_IDENTITY,
    FixtureAdapter,
    FixtureAdapterError,
    FixtureAdapterRequest,
    FixtureControl,
    FixtureTermination,
    fixture_training_state_identity,
)


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(kind: str, logical_id: str) -> RecordReference:
    return RecordReference(kind, logical_id, _identity(f"{kind}:{logical_id}"))


class _Tokenizer:
    identity = _identity("tokenizer")

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.encode())


def _adapter_request(
    tmp_path: Path,
    *,
    seed: int = 7,
    evaluation_mode: EvaluationMode = EvaluationMode.NO_QUALITY_EVALUATION,
) -> FixtureAdapterRequest:
    return _adapter_components(tmp_path, seed=seed, evaluation_mode=evaluation_mode)[0]


def _adapter_components(
    tmp_path: Path,
    *,
    seed: int = 7,
    evaluation_mode: EvaluationMode = EvaluationMode.NO_QUALITY_EVALUATION,
) -> tuple[FixtureAdapterRequest, BaseModelRevision, CompatibilityGroup]:
    prepared = DatasetService(tmp_path).import_json(
        json.dumps(
            [
                {"instruction": "Alpha", "response": "One"},
                {"instruction": "Beta", "response": "Two"},
            ],
            separators=(",", ":"),
        ).encode(),
        DatasetImportRequest(
            version_id=f"dataset-{seed}",
            field_mapping=FieldMapping("instruction", "response"),
            renderer=RendererSpec(),
            filter_rule=FilterRule(1, 100, 100),
            deduplication_rule=DeduplicationRule(),
            split_rule=SplitRule(3, (SplitPart("train", 1),)),
            tokenizer=_Tokenizer(),
        ),
    )
    model = BaseModelRevision(
        "model-fixture",
        "Fixture model",
        "fixture-family",
        "fixture-architecture",
        "public-fixture",
        "revision-one",
        _identity("weights"),
        _Tokenizer.identity,
        "Apache-2.0",
    )
    recipe = Recipe(
        "recipe-fixture",
        "fixture",
        "v1",
        "deterministic",
        "small",
        "offline",
        "none",
        "fixture",
        "periodic",
        "disabled",
        "standard",
        {},
    )
    requirements = HardwareRequirements(
        "requirements-fixture",
        ("fixture_cpu",),
        ("none",),
        0,
        1,
        ("fp32",),
        (),
        ("fixture_adapter",),
        {},
    )
    target = ExecutionTarget(
        "target-fixture",
        "fixture_cpu",
        "portable",
        "none",
        _identity("runtime-contract"),
        ("fixture_adapter",),
        {},
    )
    resolution = RecipeResolution(
        "resolution-fixture",
        record_reference(recipe),
        record_reference(model),
        record_reference(requirements),
        record_reference(target),
        "lora",
        ("k_proj", "q_proj"),
        4,
        8,
        0,
        Decimal("0.0002"),
        2,
        128,
        "fixture_adamw",
        "fp32",
        1,
        seed,
        "linear",
        4,
        2,
        "none",
        {"fixture_runtime": "v1"},
        (),
    )
    group = CompatibilityGroup(
        "group-fixture",
        record_reference(model),
        model.tokenizer_identity,
        _identity("rendering-template"),
        resolution.adapter_type,
        resolution.target_modules,
        (
            RuntimeTargetConstraint(
                target.target_class,
                target.accelerator_backend,
                target.runtime_contract,
                ("fixture_adapter",),
            ),
        ),
        (),
    )
    experiment = Experiment(
        "experiment-fixture",
        _reference("project", "project-fixture"),
        _reference("project_policy", "policy-fixture"),
        _reference("task_definition", "task-fixture"),
        prepared.version.identity,
        record_reference(model),
        model.tokenizer_identity,
        record_reference(recipe),
        record_reference(resolution),
        _identity("evaluation-policy"),
        record_reference(group),
        record_reference(requirements),
        record_reference(target),
    )
    state = fixture_training_state_identity(experiment, resolution, prepared.version, 0)
    runtime_request = ResolvedRuntimeRequest(
        "request-fixture",
        record_reference(experiment),
        experiment.manifest_identity,
        record_reference(resolution),
        prepared.version.identity,
        prepared.version.rendered_bytes_identity,
        len(prepared.rendered_bytes),
        _reference("hardware_capability_profile", "profile-fixture"),
        record_reference(target),
        FIXTURE_RUNTIME_IDENTITY,
        _identity("preflight"),
        state,
        evaluation_mode,
        resolution.training_steps,
        0,
    )
    run = Run(
        "run-fixture",
        record_reference(experiment),
        experiment.manifest_identity,
        1,
        runtime_request.hardware_capability_profile,
        record_reference(target),
        FIXTURE_RUNTIME_IDENTITY,
        runtime_request.identity,
        state,
    )
    return (
        FixtureAdapterRequest(
            experiment,
            resolution,
            prepared.version,
            prepared.rendered_bytes,
            runtime_request,
            run,
        ),
        model,
        group,
    )


def test_fixture_adapter_is_offline_repeatable_and_consumes_frozen_manifests(
    tmp_path: Path,
) -> None:
    request = _adapter_request(tmp_path / "first")
    adapter = FixtureAdapter()

    first = adapter.execute(request)
    second = adapter.execute(request)
    changed = adapter.execute(_adapter_request(tmp_path / "changed", seed=11))

    assert first == second
    assert first.termination is FixtureTermination.COMPLETED
    assert tuple(first.members) == FIXTURE_ARTIFACT_MEMBERS
    assert [item.step for item in first.progress] == [1, 2, 3, 4]
    assert [item.step for item in first.checkpoints] == [2, 4]
    assert first.members["adapter.bin"] != changed.members["adapter.bin"]
    assert b"TEMPER-FIXTURE-ADAPTER-v1" in first.members["adapter.bin"]


def test_cancellation_and_interruption_never_emit_final_artifact(
    tmp_path: Path,
) -> None:
    request = _adapter_request(tmp_path)
    adapter = FixtureAdapter()

    cancelled = adapter.execute(request, control=FixtureControl(cancel_after_step=2))
    interrupted = adapter.execute(
        request, control=FixtureControl(interrupt_after_step=3)
    )

    assert cancelled.termination is FixtureTermination.CANCELLED
    assert cancelled.members == {}
    assert cancelled.bundle_manifest is None
    assert interrupted.termination is FixtureTermination.INTERRUPTED
    assert interrupted.members == {}
    assert interrupted.bundle_manifest is None
    assert interrupted.checkpoints[-1].step == 3


def test_final_step_interruption_is_not_a_resumable_boundary(
    tmp_path: Path,
) -> None:
    request = _adapter_request(tmp_path)
    adapter = FixtureAdapter()
    final_step = request.recipe_resolution.training_steps

    with pytest.raises(FixtureAdapterError, match="fixture_control_out_of_range"):
        adapter.execute(
            request,
            control=FixtureControl(interrupt_after_step=final_step),
        )

    cancelled = adapter.execute(
        request,
        control=FixtureControl(cancel_after_step=final_step),
    )
    assert cancelled.termination is FixtureTermination.CANCELLED
    assert cancelled.checkpoints[-1].step == final_step
    assert cancelled.checkpoints[-1].resume_compatible is False
    assert cancelled.checkpoints[-1].to_receipt()["resume_compatible"] is False


def test_quality_mode_metadata_does_not_change_deterministic_adapter_bytes(
    tmp_path: Path,
) -> None:
    disabled = FixtureAdapter().execute(_adapter_request(tmp_path / "disabled"))
    future_mode = FixtureAdapter().execute(
        _adapter_request(
            tmp_path / "future",
            evaluation_mode=EvaluationMode.LIGHT_EVALUATION,
        )
    )

    assert disabled.members["adapter.bin"] == future_mode.members["adapter.bin"]
    assert disabled.bundle_manifest != future_mode.bundle_manifest
