from dataclasses import replace
from decimal import Decimal
import hashlib
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import RecordReference, record_reference
from temper_ml.runtime.paths import (
    PortableLocation,
    PortablePathError,
    WindowsWslPathMap,
)
from temper_ml.runtime.preflight import (
    EstimateComponents,
    PreflightError,
    PreflightEstimate,
    PreflightResult,
    capture_capability_profile,
    estimate_resources,
    material_change_reasons,
    preflight,
    select_execution_target,
)


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(record_type: str, logical_id: str) -> RecordReference:
    return RecordReference(
        record_type, logical_id, _identity(f"{record_type}:{logical_id}")
    )


def _target(
    target_id: str = "target-wsl2-rocm",
    target_class: str = "wsl2_rocm",
    backend: str = "rocm",
) -> ExecutionTarget:
    return ExecutionTarget(
        target_id=target_id,
        target_class=target_class,
        platform="linux" if target_class == "wsl2_rocm" else "windows",
        accelerator_backend=backend,
        runtime_contract=_identity(f"runtime:{target_class}"),
        capabilities=("lora", "local_staging"),
        constraints={"local_only": True},
    )


def _requirements() -> HardwareRequirements:
    return HardwareRequirements(
        requirements_id="requirements-balanced",
        execution_target_classes=("native_windows_rocm", "wsl2_rocm"),
        accelerator_backends=("rocm",),
        minimum_accelerator_memory_bytes=4_000,
        minimum_system_memory_bytes=8_000,
        required_precision_modes=("bf16",),
        required_quantization_modes=(),
        required_capabilities=("lora",),
        constraints={"local_only": True},
    )


def _resolution(
    requirements: HardwareRequirements, target: ExecutionTarget
) -> RecipeResolution:
    recipe = Recipe(
        recipe_id="recipe-balanced",
        family="balanced",
        version="v1",
        training_profile="balanced",
        adapter_size="small",
        memory_mode="standard",
        quantization="none",
        training_duration="fixture",
        checkpoint_policy="periodic",
        evaluation_intensity="normal",
        retention_policy="standard",
        expert_overrides={},
    )
    return RecipeResolution(
        resolution_id=f"resolution-{target.target_id}",
        recipe=record_reference(recipe),
        base_model_revision=_reference("base_model_revision", "model-synthetic"),
        hardware_requirements=record_reference(requirements),
        execution_target=record_reference(target),
        adapter_type="lora",
        target_modules=("k_proj", "q_proj"),
        rank=8,
        alpha=16,
        dropout=0,
        learning_rate=Decimal("0.0002"),
        effective_batch_size=8,
        sequence_length=512,
        optimizer="adamw",
        precision="bf16",
        gradient_accumulation=2,
        seed=7,
        schedule="linear",
        training_steps=20,
        checkpoint_cadence=5,
        quantization="none",
        library_versions={"fixture_runtime": "v1"},
        applied_constraints=(),
    )


def _profile(
    target: ExecutionTarget,
    memory: int = 8_000,
    runtime_version: str = "v1",
):
    return capture_capability_profile(
        profile_id=f"profile-{memory}",
        execution_target=target,
        accelerator_backend="rocm",
        accelerator_architecture="synthetic-arch",
        accelerator_model="Synthetic accelerator",
        accelerator_count=1,
        accelerator_memory_bytes=(memory,),
        system_memory_bytes=16_000,
        supported_precision_modes=("bf16",),
        supported_quantization_modes=(),
        capabilities=("lora", "local_staging"),
        library_versions={"fixture_runtime": runtime_version},
    )


def _components() -> EstimateComponents:
    return EstimateComponents(
        base_model_bytes=2_000,
        adapter_optimizer_bytes=500,
        peak_activation_bytes=1_000,
        accelerator_runtime_overhead_bytes=500,
        dataset_bytes=1_000,
        host_runtime_overhead_bytes=2_000,
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("accelerator_memory_bytes", -1, "accelerator_memory_bytes_invalid"),
        ("system_memory_bytes", -1, "system_memory_bytes_invalid"),
        ("accelerator_memory_bytes", True, "accelerator_memory_bytes_invalid"),
        ("training_steps", 0, "training_steps_invalid"),
        ("training_steps", -1, "training_steps_invalid"),
        ("effective_batch_size", 0, "effective_batch_size_invalid"),
        ("effective_batch_size", -1, "effective_batch_size_invalid"),
        ("effective_batch_size", True, "effective_batch_size_invalid"),
    ),
)
def test_preflight_estimate_rejects_invalid_resource_dimensions(
    field: str, value: object, code: str
) -> None:
    values: dict[str, object] = {
        "accelerator_memory_bytes": 4_000,
        "system_memory_bytes": 5_000,
        "training_steps": 20,
        "effective_batch_size": 8,
    }
    values[field] = value

    with pytest.raises(PreflightError) as error:
        PreflightEstimate(**values)  # type: ignore[arg-type]

    assert error.value.code == code


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("training_steps", 21, "estimate_training_steps_mismatch"),
        (
            "effective_batch_size",
            9,
            "estimate_effective_batch_size_mismatch",
        ),
    ),
)
def test_preflight_rejects_estimates_inconsistent_with_resolution(
    field: str, value: int, code: str
) -> None:
    requirements = _requirements()
    target = _target()
    resolution = _resolution(requirements, target)
    profile = _profile(target)
    estimate = replace(
        estimate_resources(resolution, _components()),
        **{field: value},
    )

    with pytest.raises(PreflightError) as preflight_error:
        preflight(resolution, requirements, target, profile, estimate)
    with pytest.raises(PreflightError) as result_error:
        PreflightResult(
            resolution,
            requirements,
            target,
            profile,
            estimate,
            (),
        )

    assert preflight_error.value.code == code
    assert result_error.value.code == code


def test_preflight_exposes_every_estimate_and_constraint() -> None:
    requirements = _requirements()
    target = _target()
    resolution = _resolution(requirements, target)
    estimate = estimate_resources(resolution, _components())

    ready = preflight(resolution, requirements, target, _profile(target), estimate)
    blocked = preflight(
        resolution, requirements, target, _profile(target, 3_000), estimate
    )

    assert ready.ready
    assert ready.to_view()["estimate"] == {
        "accelerator_memory_bytes": 4_000,
        "system_memory_bytes": 5_000,
        "training_steps": 20,
        "effective_batch_size": 8,
    }
    assert not blocked.ready
    assert blocked.blocking_reasons == (
        "minimum_accelerator_memory",
        "estimated_accelerator_memory",
    )

    version_drift = preflight(
        resolution,
        requirements,
        target,
        _profile(target, runtime_version="v2"),
        estimate,
    )
    assert version_drift.blocking_reasons == ("resolved_library_versions_match",)


def test_preflight_result_rejects_missing_or_altered_canonical_checks() -> None:
    requirements = _requirements()
    target = _target()
    resolution = _resolution(requirements, target)
    profile = _profile(target, 3_000)
    estimate = estimate_resources(resolution, _components())
    blocked = preflight(resolution, requirements, target, profile, estimate)

    assert not blocked.ready
    forged_checks = (
        (),
        blocked.checks[:-1],
        (*blocked.checks, blocked.checks[-1]),
        tuple(reversed(blocked.checks)),
        (replace(blocked.checks[0], satisfied=False), *blocked.checks[1:]),
    )
    for checks in forged_checks:
        with pytest.raises(PreflightError) as error:
            PreflightResult(
                resolution,
                requirements,
                target,
                profile,
                estimate,
                checks,
            )
        assert error.value.code == "preflight_checks_mismatch"


def test_preflight_result_direct_construction_is_deterministic() -> None:
    requirements = _requirements()
    target = _target()
    resolution = _resolution(requirements, target)
    estimate = estimate_resources(resolution, _components())

    for profile in (_profile(target), _profile(target, 3_000)):
        canonical = preflight(
            resolution,
            requirements,
            target,
            profile,
            estimate,
        )
        direct = PreflightResult(
            resolution,
            requirements,
            target,
            profile,
            estimate,
            canonical.checks,
        )
        assert direct == canonical
        assert direct.ready is canonical.ready
        assert direct.blocking_reasons == canonical.blocking_reasons


@pytest.mark.parametrize(
    ("context", "code"),
    (
        ("requirements", "resolution_requirements_mismatch"),
        ("target", "resolution_target_mismatch"),
        ("profile", "profile_target_mismatch"),
    ),
)
def test_preflight_result_rejects_direct_reference_mismatches(
    context: str, code: str
) -> None:
    requirements = _requirements()
    target = _target()
    resolution = _resolution(requirements, target)
    profile = _profile(target)
    estimate = estimate_resources(resolution, _components())
    canonical = preflight(
        resolution,
        requirements,
        target,
        profile,
        estimate,
    )

    if context == "requirements":
        requirements = replace(
            requirements,
            requirements_id="requirements-other",
        )
    elif context == "target":
        target = _target("target-other")
    else:
        profile = _profile(_target("target-other"))

    with pytest.raises(PreflightError) as error:
        PreflightResult(
            resolution,
            requirements,
            target,
            profile,
            estimate,
            canonical.checks,
        )

    assert error.value.code == code


def test_target_selection_never_silently_switches_between_windows_and_wsl() -> None:
    requirements = _requirements()
    wsl = _target()
    native = _target("target-native-rocm", "native_windows_rocm")

    with pytest.raises(PreflightError, match="execution_target_selection_required"):
        select_execution_target(requirements, (native, wsl))

    assert (
        select_execution_target(
            requirements,
            (native, wsl),
            selected_target_id=wsl.target_id,
        )
        == wsl
    )


def test_material_change_rules_ignore_machine_identity_but_expose_platform_change() -> (
    None
):
    requirements = _requirements()
    original_target = _target()
    original_resolution = _resolution(requirements, original_target)
    assert (
        material_change_reasons(
            original_target,
            original_target,
            original_resolution,
            original_resolution,
        )
        == ()
    )

    native = _target("target-native-rocm", "native_windows_rocm")
    changed_resolution = replace(
        original_resolution,
        resolution_id="resolution-native",
        execution_target=record_reference(native),
    )
    reasons = material_change_reasons(
        original_target, native, original_resolution, changed_resolution
    )
    assert reasons == (
        "execution_target_class_changed",
        "execution_target_revision_changed",
        "recipe_resolution_changed",
    )


def test_portable_path_mapping_keeps_host_and_worker_roots_out_of_manifest_values() -> (
    None
):
    mapping = WindowsWslPathMap(
        PureWindowsPath("X:/synthetic-project"),
        PurePosixPath("/mnt/synthetic-project"),
    )
    location = mapping.portable_from_host(
        PureWindowsPath("X:/synthetic-project/staging/dataset.jsonl")
    )

    assert location.to_dict() == {"logical_path": "staging/dataset.jsonl"}
    assert mapping.host_path(location) == PureWindowsPath(
        "X:/synthetic-project/staging/dataset.jsonl"
    )
    assert mapping.worker_path(location) == PurePosixPath(
        "/mnt/synthetic-project/staging/dataset.jsonl"
    )
    with pytest.raises(PortablePathError, match="portable_location_invalid"):
        PortableLocation("../outside")
    with pytest.raises(PortablePathError, match="portable_location_invalid"):
        PortableLocation("X:/synthetic-project/model.bin")
    with pytest.raises(PortablePathError, match="host_path_outside_project"):
        mapping.portable_from_host(PureWindowsPath("Y:/outside-project/source.bin"))


def test_capability_capture_rejects_private_facts_with_a_stable_error() -> None:
    with pytest.raises(PreflightError, match="capability_profile_invalid"):
        capture_capability_profile(
            profile_id="profile-private",
            execution_target=_target(),
            accelerator_backend="rocm",
            accelerator_architecture="synthetic-arch",
            accelerator_model="Synthetic accelerator",
            accelerator_count=1,
            accelerator_memory_bytes=(8_000,),
            system_memory_bytes=16_000,
            supported_precision_modes=("bf16",),
            supported_quantization_modes=(),
            capabilities=("lora",),
            library_versions={"hostname": "private-machine"},
        )
