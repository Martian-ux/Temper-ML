from decimal import Decimal
import hashlib

import pytest

from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import Recipe
from temper_ml.runtime.recipe_resolution import (
    RecipeCatalog,
    RecipeCatalogEntry,
    RecipeResolutionError,
    RecipeResolver,
    ResolutionConstraint,
)


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _recipe(overrides=None) -> Recipe:
    return Recipe(
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
        expert_overrides=overrides or {},
    )


def _defaults() -> dict[str, object]:
    return {
        "adapter_type": "lora",
        "target_modules": ["k_proj", "q_proj"],
        "rank": 8,
        "alpha": 16,
        "dropout": 0,
        "learning_rate": Decimal("0.0002"),
        "effective_batch_size": 8,
        "sequence_length": 512,
        "optimizer": "adamw",
        "precision": "bf16",
        "gradient_accumulation": 2,
        "seed": 7,
        "schedule": "linear",
        "training_steps": 20,
        "checkpoint_cadence": 5,
        "quantization": "none",
        "library_versions": {"fixture_runtime": "v1"},
    }


def _model() -> BaseModelRevision:
    return BaseModelRevision(
        model_id="model-synthetic",
        display_name="Synthetic model",
        model_family="fixture-family",
        architecture="fixture-causal-lm",
        source="public-fixture",
        revision="revision-one",
        weights_identity=_identity("weights"),
        tokenizer_identity=_identity("tokenizer"),
        license="Apache-2.0",
    )


def _requirements() -> HardwareRequirements:
    return HardwareRequirements(
        requirements_id="requirements-balanced",
        execution_target_classes=("wsl2_rocm",),
        accelerator_backends=("rocm",),
        minimum_accelerator_memory_bytes=4_000,
        minimum_system_memory_bytes=8_000,
        required_precision_modes=("bf16",),
        required_quantization_modes=(),
        required_capabilities=("lora",),
        constraints={"local_only": True},
    )


def _target() -> ExecutionTarget:
    return ExecutionTarget(
        target_id="target-wsl2-rocm",
        target_class="wsl2_rocm",
        platform="linux",
        accelerator_backend="rocm",
        runtime_contract=_identity("runtime-contract"),
        capabilities=("lora",),
        constraints={"local_only": True},
    )


def test_catalog_overrides_and_constraints_resolve_deterministically() -> None:
    catalog = RecipeCatalog(
        (
            RecipeCatalogEntry(
                _recipe(),
                _defaults(),
                ("rank", "sequence_length"),
            ),
        )
    )
    entry = catalog.with_expert_overrides(
        "balanced", "v1", {"rank": 16, "sequence_length": 768}
    )
    constraints = (
        ResolutionConstraint("memory_budget", {"gradient_accumulation": 4}),
        ResolutionConstraint("platform_precision", {"precision": "bf16"}),
    )
    resolver = RecipeResolver()

    first = resolver.resolve(
        entry,
        base_model_revision=_model(),
        hardware_requirements=_requirements(),
        execution_target=_target(),
        constraints=constraints,
    )
    second = resolver.resolve(
        entry,
        base_model_revision=_model(),
        hardware_requirements=_requirements(),
        execution_target=_target(),
        constraints=reversed(constraints),
    )

    assert first.identity == second.identity
    assert first.resolution_id == second.resolution_id
    assert first.rank == 16
    assert first.sequence_length == 768
    assert first.gradient_accumulation == 4
    assert first.applied_constraints == ("memory_budget", "platform_precision")
    assert entry.recipe.to_payload()["expert_overrides"] == {
        "rank": 16,
        "sequence_length": 768,
    }


def test_catalog_rejects_hidden_or_unknown_expert_inputs() -> None:
    with pytest.raises(RecipeResolutionError, match="expert_override_not_allowed"):
        RecipeCatalogEntry(_recipe({"learning_rate": Decimal("0.001")}), _defaults())

    catalog = RecipeCatalog((RecipeCatalogEntry(_recipe(), _defaults(), ("rank",)),))
    with pytest.raises(RecipeResolutionError, match="expert_override_not_allowed"):
        catalog.with_expert_overrides("balanced", "v1", {"checkpoint_cadence": 1})


def test_constraint_conflicts_and_catalog_duplicates_fail_closed() -> None:
    entry = RecipeCatalogEntry(_recipe(), _defaults())
    with pytest.raises(RecipeResolutionError, match="recipe_catalog_duplicate"):
        RecipeCatalog((entry, entry))

    with pytest.raises(RecipeResolutionError, match="constraint_conflict"):
        RecipeResolver().resolve(
            entry,
            base_model_revision=_model(),
            hardware_requirements=_requirements(),
            execution_target=_target(),
            constraints=(
                ResolutionConstraint("first", {"rank": 8}),
                ResolutionConstraint("second", {"rank": 16}),
            ),
        )
