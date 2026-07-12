"""Versioned recipe catalog and deterministic technical resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Any, Iterable, Mapping

from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.hardware import ExecutionTarget, HardwareRequirements
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import (
    RecordValidationError,
    freeze_json_object,
    record_reference,
    thaw_json,
)
from temper_ml.store.canonical_json import CanonicalJsonError, dumps_canonical_json


class RecipeResolutionError(RuntimeError):
    """A stable recipe-catalog or resolution failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TECHNICAL_FIELDS = (
    "adapter_type",
    "target_modules",
    "rank",
    "alpha",
    "dropout",
    "learning_rate",
    "effective_batch_size",
    "sequence_length",
    "optimizer",
    "precision",
    "gradient_accumulation",
    "seed",
    "schedule",
    "training_steps",
    "checkpoint_cadence",
    "quantization",
    "library_versions",
)
_TECHNICAL_FIELD_SET = frozenset(_TECHNICAL_FIELDS)


@dataclass(frozen=True)
class ResolutionConstraint:
    """One explicit hardware or platform constraint and its visible changes."""

    code: str
    changes: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or _IDENTIFIER.fullmatch(self.code) is None:
            raise RecipeResolutionError("constraint_code_invalid")
        try:
            frozen = freeze_json_object(self.changes, field="constraint changes")
        except (CanonicalJsonError, RecordValidationError, TypeError, ValueError):
            raise RecipeResolutionError("constraint_changes_invalid") from None
        if not frozen or set(frozen) - _TECHNICAL_FIELD_SET:
            raise RecipeResolutionError("constraint_changes_invalid")
        object.__setattr__(self, "changes", frozen)


@dataclass(frozen=True)
class RecipeCatalogEntry:
    """One catalog recipe with complete, inspectable technical defaults."""

    recipe: Recipe
    technical_defaults: Mapping[str, Any]
    allowed_expert_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, Recipe):
            raise RecipeResolutionError("catalog_recipe_invalid")
        try:
            defaults = freeze_json_object(
                self.technical_defaults, field="technical defaults"
            )
        except (CanonicalJsonError, RecordValidationError, TypeError, ValueError):
            raise RecipeResolutionError("catalog_defaults_invalid") from None
        if set(defaults) != _TECHNICAL_FIELD_SET:
            raise RecipeResolutionError("catalog_defaults_incomplete")
        if not isinstance(self.allowed_expert_overrides, tuple) or any(
            not isinstance(field, str) or field not in _TECHNICAL_FIELD_SET
            for field in self.allowed_expert_overrides
        ):
            raise RecipeResolutionError("catalog_override_allowlist_invalid")
        allowed = tuple(sorted(set(self.allowed_expert_overrides)))
        if len(allowed) != len(self.allowed_expert_overrides):
            raise RecipeResolutionError("catalog_override_allowlist_invalid")
        if set(self.recipe.expert_overrides) - set(allowed):
            raise RecipeResolutionError("expert_override_not_allowed")
        object.__setattr__(self, "technical_defaults", defaults)
        object.__setattr__(self, "allowed_expert_overrides", allowed)


class RecipeCatalog:
    """An immutable catalog selected by exact family and version."""

    def __init__(self, entries: Iterable[RecipeCatalogEntry]) -> None:
        values = tuple(entries)
        if not values or any(
            not isinstance(entry, RecipeCatalogEntry) for entry in values
        ):
            raise RecipeResolutionError("recipe_catalog_invalid")
        keys = tuple((entry.recipe.family, entry.recipe.version) for entry in values)
        if len(set(keys)) != len(keys):
            raise RecipeResolutionError("recipe_catalog_duplicate")
        self._entries = tuple(
            sorted(
                values, key=lambda entry: (entry.recipe.family, entry.recipe.version)
            )
        )

    @property
    def entries(self) -> tuple[RecipeCatalogEntry, ...]:
        return self._entries

    def select(self, family: str, version: str) -> RecipeCatalogEntry:
        matches = tuple(
            entry
            for entry in self._entries
            if entry.recipe.family == family and entry.recipe.version == version
        )
        if not matches:
            raise RecipeResolutionError("recipe_not_found")
        return matches[0]

    def with_expert_overrides(
        self,
        family: str,
        version: str,
        overrides: Mapping[str, Any],
    ) -> RecipeCatalogEntry:
        """Create an exact recipe revision whose overrides remain explicit inputs."""

        entry = self.select(family, version)
        try:
            recipe = replace(entry.recipe, expert_overrides=overrides)
        except (CanonicalJsonError, RecordValidationError, TypeError, ValueError):
            raise RecipeResolutionError("expert_overrides_invalid") from None
        return RecipeCatalogEntry(
            recipe,
            entry.technical_defaults,
            entry.allowed_expert_overrides,
        )

    def to_view(self) -> list[dict[str, object]]:
        return [
            {
                "recipe_id": entry.recipe.recipe_id,
                "family": entry.recipe.family,
                "version": entry.recipe.version,
                "allowed_expert_overrides": list(entry.allowed_expert_overrides),
            }
            for entry in self._entries
        ]


class RecipeResolver:
    """Resolve defaults, explicit expert inputs, and visible constraints."""

    def resolve(
        self,
        entry: RecipeCatalogEntry,
        *,
        base_model_revision: BaseModelRevision,
        hardware_requirements: HardwareRequirements,
        execution_target: ExecutionTarget,
        constraints: Iterable[ResolutionConstraint] = (),
    ) -> RecipeResolution:
        if not isinstance(entry, RecipeCatalogEntry):
            raise RecipeResolutionError("catalog_entry_invalid")
        if not isinstance(base_model_revision, BaseModelRevision):
            raise RecipeResolutionError("base_model_revision_invalid")
        if not isinstance(hardware_requirements, HardwareRequirements):
            raise RecipeResolutionError("hardware_requirements_invalid")
        if not isinstance(execution_target, ExecutionTarget):
            raise RecipeResolutionError("execution_target_invalid")

        defaults = thaw_json(entry.technical_defaults)
        if not isinstance(defaults, dict):
            raise RecipeResolutionError("catalog_defaults_invalid")
        settings: dict[str, Any] = dict(defaults)
        overrides = thaw_json(entry.recipe.expert_overrides)
        if not isinstance(overrides, dict):
            raise RecipeResolutionError("expert_overrides_invalid")
        if set(overrides) - set(entry.allowed_expert_overrides):
            raise RecipeResolutionError("expert_override_not_allowed")
        for field in sorted(overrides):
            settings[field] = overrides[field]

        constraint_values: dict[str, Any] = {}
        constraint_items = tuple(constraints)
        if any(
            not isinstance(constraint, ResolutionConstraint)
            for constraint in constraint_items
        ):
            raise RecipeResolutionError("constraint_invalid")
        ordered_constraints = tuple(
            sorted(constraint_items, key=lambda constraint: constraint.code)
        )
        codes = tuple(constraint.code for constraint in ordered_constraints)
        if len(set(codes)) != len(codes):
            raise RecipeResolutionError("constraint_duplicate")
        for constraint in ordered_constraints:
            changes = thaw_json(constraint.changes)
            if not isinstance(changes, dict):
                raise RecipeResolutionError("constraint_changes_invalid")
            for field in sorted(changes):
                value = changes[field]
                if field in constraint_values and dumps_canonical_json(
                    constraint_values[field]
                ) != dumps_canonical_json(value):
                    raise RecipeResolutionError("constraint_conflict")
                constraint_values[field] = value
                settings[field] = value

        if set(settings) != _TECHNICAL_FIELD_SET:
            raise RecipeResolutionError("resolved_settings_incomplete")
        target_modules = settings["target_modules"]
        if isinstance(target_modules, list):
            settings["target_modules"] = tuple(target_modules)
        recipe_reference = record_reference(entry.recipe)
        model_reference = record_reference(base_model_revision)
        requirements_reference = record_reference(hardware_requirements)
        target_reference = record_reference(execution_target)
        identity_input = {
            "recipe": recipe_reference.to_dict(),
            "base_model_revision": model_reference.to_dict(),
            "hardware_requirements": requirements_reference.to_dict(),
            "execution_target": target_reference.to_dict(),
            **settings,
            "applied_constraints": list(codes),
        }
        try:
            digest = hashlib.sha256(dumps_canonical_json(identity_input)).hexdigest()
            return RecipeResolution(
                resolution_id=f"resolution-{digest}",
                recipe=recipe_reference,
                base_model_revision=model_reference,
                hardware_requirements=requirements_reference,
                execution_target=target_reference,
                applied_constraints=codes,
                **settings,
            )
        except (CanonicalJsonError, RecordValidationError, TypeError, ValueError):
            raise RecipeResolutionError("recipe_resolution_invalid") from None


def resolution_view(resolution: RecipeResolution) -> dict[str, object]:
    """Return an exact local view of a resolved immutable manifest."""

    if not isinstance(resolution, RecipeResolution):
        raise RecipeResolutionError("recipe_resolution_invalid")
    return {
        "status": "resolved",
        "identity": {
            "algorithm": resolution.identity.algorithm,
            "value": resolution.identity.value,
        },
        "manifest": resolution.to_payload(),
    }
