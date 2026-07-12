"""Versioned recipe and fully resolved training-manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar, Mapping

from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    freeze_json_object,
    require_identifier,
    require_non_negative_int,
    require_positive_int,
    require_string_tuple,
    require_text,
    thaw_json,
)
from temper_ml.store.canonical_json import CanonicalJsonError, dumps_canonical_json


def _require_reference(
    field: str, value: RecordReference, record_type: str
) -> RecordReference:
    if not isinstance(value, RecordReference) or value.record_type != record_type:
        raise RecordValidationError(f"{field} must reference {record_type}")
    return value


def _require_decimal(
    field: str,
    value: Decimal,
    *,
    minimum: Decimal,
    maximum_exclusive: Decimal | None = None,
) -> Decimal:
    if not isinstance(value, Decimal):
        raise RecordValidationError(f"{field} must be a normalized Decimal")
    try:
        dumps_canonical_json({field: value})
    except CanonicalJsonError as exc:
        raise RecordValidationError(f"{field} must be a normalized Decimal") from exc
    if value < minimum or (
        maximum_exclusive is not None and value >= maximum_exclusive
    ):
        raise RecordValidationError(f"{field} is outside its supported range")
    return value


def _require_dropout(field: str, value: Decimal | int) -> Decimal | int:
    if isinstance(value, int) and not isinstance(value, bool) and value == 0:
        return value
    if not isinstance(value, Decimal):
        raise RecordValidationError(f"{field} must be zero or a normalized Decimal")
    return _require_decimal(
        field,
        value,
        minimum=Decimal("0.0"),
        maximum_exclusive=Decimal("1.0"),
    )


@dataclass(frozen=True)
class Recipe(TypedRecord):
    """Versioned, user-facing choices plus explicit expert overrides."""

    RECORD_TYPE: ClassVar[str] = "recipe"

    recipe_id: str
    family: str
    version: str
    training_profile: str
    adapter_size: str
    memory_mode: str
    quantization: str
    training_duration: str
    checkpoint_policy: str
    evaluation_intensity: str
    retention_policy: str
    expert_overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_identifier("recipe_id", self.recipe_id)
        require_identifier("family", self.family)
        require_identifier("version", self.version)
        for field in (
            "training_profile",
            "adapter_size",
            "memory_mode",
            "quantization",
            "training_duration",
            "checkpoint_policy",
            "evaluation_intensity",
            "retention_policy",
        ):
            require_text(field, getattr(self, field))
        object.__setattr__(
            self,
            "expert_overrides",
            freeze_json_object(self.expert_overrides, field="expert_overrides"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "family": self.family,
            "version": self.version,
            "training_profile": self.training_profile,
            "adapter_size": self.adapter_size,
            "memory_mode": self.memory_mode,
            "quantization": self.quantization,
            "training_duration": self.training_duration,
            "checkpoint_policy": self.checkpoint_policy,
            "evaluation_intensity": self.evaluation_intensity,
            "retention_policy": self.retention_policy,
            "expert_overrides": thaw_json(self.expert_overrides),
        }


@dataclass(frozen=True)
class RecipeResolution(TypedRecord):
    """Complete immutable technical settings resolved from a recipe."""

    RECORD_TYPE: ClassVar[str] = "recipe_resolution"

    resolution_id: str
    recipe: RecordReference
    base_model_revision: RecordReference
    hardware_requirements: RecordReference
    execution_target: RecordReference
    adapter_type: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    dropout: Decimal | int
    learning_rate: Decimal
    effective_batch_size: int
    sequence_length: int
    optimizer: str
    precision: str
    gradient_accumulation: int
    seed: int
    schedule: str
    training_steps: int
    checkpoint_cadence: int
    quantization: str
    library_versions: Mapping[str, Any]
    applied_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier("resolution_id", self.resolution_id)
        _require_reference("recipe", self.recipe, "recipe")
        _require_reference(
            "base_model_revision", self.base_model_revision, "base_model_revision"
        )
        _require_reference(
            "hardware_requirements", self.hardware_requirements, "hardware_requirements"
        )
        _require_reference(
            "execution_target", self.execution_target, "execution_target"
        )
        require_identifier("adapter_type", self.adapter_type)
        object.__setattr__(
            self,
            "target_modules",
            require_string_tuple(
                "target_modules", self.target_modules, sorted_values=True
            ),
        )
        require_positive_int("rank", self.rank)
        require_positive_int("alpha", self.alpha)
        _require_dropout("dropout", self.dropout)
        _require_decimal(
            "learning_rate", self.learning_rate, minimum=Decimal("0.000000000000000001")
        )
        for field in (
            "effective_batch_size",
            "sequence_length",
            "gradient_accumulation",
            "training_steps",
            "checkpoint_cadence",
        ):
            require_positive_int(field, getattr(self, field))
        require_non_negative_int("seed", self.seed)
        for field in ("optimizer", "precision", "schedule", "quantization"):
            require_identifier(field, getattr(self, field))
        object.__setattr__(
            self,
            "library_versions",
            freeze_json_object(self.library_versions, field="library_versions"),
        )
        object.__setattr__(
            self,
            "applied_constraints",
            require_string_tuple(
                "applied_constraints",
                self.applied_constraints,
                non_empty=False,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "resolution_id": self.resolution_id,
            "recipe": self.recipe.to_dict(),
            "base_model_revision": self.base_model_revision.to_dict(),
            "hardware_requirements": self.hardware_requirements.to_dict(),
            "execution_target": self.execution_target.to_dict(),
            "adapter_type": self.adapter_type,
            "target_modules": list(self.target_modules),
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "effective_batch_size": self.effective_batch_size,
            "sequence_length": self.sequence_length,
            "optimizer": self.optimizer,
            "precision": self.precision,
            "gradient_accumulation": self.gradient_accumulation,
            "seed": self.seed,
            "schedule": self.schedule,
            "training_steps": self.training_steps,
            "checkpoint_cadence": self.checkpoint_cadence,
            "quantization": self.quantization,
            "library_versions": thaw_json(self.library_versions),
            "applied_constraints": list(self.applied_constraints),
        }
