from decimal import Decimal
import hashlib

import pytest

from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    record_reference,
)


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(kind: str, logical_id: str, revision: str = "v1") -> RecordReference:
    return RecordReference(
        kind, logical_id, _identity(f"{kind}:{logical_id}:{revision}")
    )


def _target() -> ExecutionTarget:
    return ExecutionTarget(
        target_id="target-wsl2-rocm",
        target_class="wsl2_rocm",
        platform="linux",
        accelerator_backend="rocm",
        runtime_contract=_identity("runtime-contract"),
        capabilities=("lora", "bf16"),
        constraints={"minimum_runtime_version": "v1"},
    )


def _requirements() -> HardwareRequirements:
    return HardwareRequirements(
        requirements_id="requirements-balanced",
        execution_target_classes=("wsl2_rocm",),
        accelerator_backends=("rocm",),
        minimum_accelerator_memory_bytes=8_000_000_000,
        minimum_system_memory_bytes=16_000_000_000,
        required_precision_modes=("bf16",),
        required_quantization_modes=(),
        required_capabilities=("lora",),
        constraints={"maximum_sequence_length": 2048},
    )


def _recipe(overrides=None) -> Recipe:
    return Recipe(
        recipe_id="recipe-balanced-v1",
        family="balanced",
        version="v1",
        training_profile="balanced",
        adapter_size="small",
        memory_mode="standard",
        quantization="none",
        training_duration="short",
        checkpoint_policy="periodic",
        evaluation_intensity="light",
        retention_policy="full",
        expert_overrides=overrides or {},
    )


def _resolution(
    recipe: Recipe, target: ExecutionTarget, requirements: HardwareRequirements
) -> RecipeResolution:
    return RecipeResolution(
        resolution_id="resolution-balanced-v1",
        recipe=record_reference(recipe, recipe.recipe_id),
        base_model_revision=_reference("base_model_revision", "model-alpha"),
        hardware_requirements=record_reference(
            requirements, requirements.requirements_id
        ),
        execution_target=record_reference(target, target.target_id),
        adapter_type="lora",
        target_modules=("v_proj", "q_proj"),
        rank=8,
        alpha=16,
        dropout=Decimal("0.05"),
        learning_rate=Decimal("0.0002"),
        effective_batch_size=8,
        sequence_length=1024,
        optimizer="adamw",
        precision="bf16",
        gradient_accumulation=4,
        seed=7,
        schedule="cosine",
        training_steps=100,
        checkpoint_cadence=25,
        quantization="none",
        library_versions={"transformers": "v1", "peft": "v1"},
        applied_constraints=("memory_budget",),
    )


def test_recipe_resolution_is_complete_deterministic_and_order_normalized() -> None:
    recipe = _recipe()
    target = _target()
    requirements = _requirements()
    first = _resolution(recipe, target, requirements)
    second = _resolution(recipe, target, requirements)

    assert first.identity == second.identity
    assert first.target_modules == ("q_proj", "v_proj")
    assert first.to_payload()["seed"] == 7
    assert first.to_payload()["library_versions"] == {
        "peft": "v1",
        "transformers": "v1",
    }
    zero_dropout = RecipeResolution(**{**first.__dict__, "dropout": 0})
    assert zero_dropout.to_payload()["dropout"] == 0


def test_recipe_overrides_are_explicit_and_alias_safe() -> None:
    overrides = {"rank": 16, "nested": {"enabled": True}}
    recipe = _recipe(overrides)
    before = recipe.identity

    overrides["rank"] = 32
    overrides["nested"]["enabled"] = False

    assert recipe.identity == before
    assert recipe.to_payload()["expert_overrides"] == {
        "nested": {"enabled": True},
        "rank": 16,
    }


def test_capability_profile_rejects_private_operational_facts() -> None:
    target = _target()
    common = dict(
        profile_id="profile-synthetic",
        execution_target=record_reference(target, target.target_id),
        accelerator_backend="rocm",
        accelerator_architecture="synthetic-arch",
        accelerator_model="Synthetic GPU",
        accelerator_count=1,
        accelerator_memory_bytes=(16_000_000_000,),
        system_memory_bytes=32_000_000_000,
        supported_precision_modes=("bf16",),
        supported_quantization_modes=(),
        capabilities=("lora",),
    )

    with pytest.raises(RecordValidationError, match="private capability key"):
        HardwareCapabilityProfile(
            library_versions={"hostname": "private-machine"}, **common
        )
    with pytest.raises(RecordValidationError, match="non-public location"):
        HardwareCapabilityProfile(
            library_versions={"runtime": "C:\\private\\runtime"}, **common
        )

    fixture_profile = HardwareCapabilityProfile(
        **{
            **common,
            "profile_id": "profile-fixture",
            "accelerator_backend": "fixture",
            "accelerator_architecture": "none",
            "accelerator_model": "No accelerator",
            "accelerator_count": 0,
            "accelerator_memory_bytes": (),
            "library_versions": {"fixture_runtime": "v1"},
        }
    )
    assert fixture_profile.to_payload()["accelerator_count"] == 0


def test_resolution_rejects_float_and_non_normalized_decimal_inputs() -> None:
    recipe = _recipe()
    target = _target()
    requirements = _requirements()
    with pytest.raises((RecordValidationError, ValueError), match="Decimal|float"):
        RecipeResolution(
            **{
                **_resolution(recipe, target, requirements).__dict__,
                "learning_rate": 0.0002,
            }
        )
    with pytest.raises(RecordValidationError, match="normalized Decimal"):
        RecipeResolution(
            **{
                **_resolution(recipe, target, requirements).__dict__,
                "learning_rate": Decimal("0.00020"),
            }
        )
