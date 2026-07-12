"""Independent baseline comparison policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    identity_fields,
    require_identifier,
)


class BaselineKind(str, Enum):
    PER_MODEL = "per_model"
    PROJECT_CHAMPION = "project_champion"
    FIXED_REFERENCE = "fixed_reference"


@dataclass(frozen=True)
class PerModelBaseline:
    """Compare a candidate with its own exact base-model revision."""

    comparison_policy: ContentIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.comparison_policy, ContentIdentity):
            raise RecordValidationError(
                "per-model comparison_policy must be a content identity"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": BaselineKind.PER_MODEL.value,
            "comparison_policy": identity_fields(self.comparison_policy),
        }


@dataclass(frozen=True)
class ProjectChampionBaseline:
    """Resolve the project champion independently at comparison time."""

    comparison_policy: ContentIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.comparison_policy, ContentIdentity):
            raise RecordValidationError(
                "project-champion comparison_policy must be a content identity"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": BaselineKind.PROJECT_CHAMPION.value,
            "comparison_policy": identity_fields(self.comparison_policy),
        }


@dataclass(frozen=True)
class FixedReferenceBaseline:
    """Compare against one stable, policy-pinned record revision."""

    comparison_policy: ContentIdentity
    reference: RecordReference

    def __post_init__(self) -> None:
        if not isinstance(self.comparison_policy, ContentIdentity):
            raise RecordValidationError(
                "fixed-reference comparison_policy must be a content identity"
            )
        if not isinstance(self.reference, RecordReference):
            raise RecordValidationError(
                "fixed-reference baseline requires a pinned record reference"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": BaselineKind.FIXED_REFERENCE.value,
            "comparison_policy": identity_fields(self.comparison_policy),
            "reference": self.reference.to_dict(),
        }


BaselineRule = PerModelBaseline | ProjectChampionBaseline | FixedReferenceBaseline


@dataclass(frozen=True)
class BaselinePolicy(TypedRecord):
    """An immutable set of independent, explicitly selected baseline rules."""

    RECORD_TYPE: ClassVar[str] = "baseline_policy"

    policy_id: str
    rules: tuple[BaselineRule, ...]

    def __post_init__(self) -> None:
        require_identifier("policy_id", self.policy_id)
        if not isinstance(self.rules, tuple) or not self.rules:
            raise RecordValidationError("baseline rules must be a non-empty tuple")
        kinds: list[str] = []
        for rule in self.rules:
            if isinstance(rule, PerModelBaseline):
                kinds.append(BaselineKind.PER_MODEL.value)
            elif isinstance(rule, ProjectChampionBaseline):
                kinds.append(BaselineKind.PROJECT_CHAMPION.value)
            elif isinstance(rule, FixedReferenceBaseline):
                kinds.append(BaselineKind.FIXED_REFERENCE.value)
            else:
                raise RecordValidationError("unsupported baseline rule")
        if len(set(kinds)) != len(kinds):
            raise RecordValidationError("baseline rule kinds must be unique")
        ordered = tuple(rule for _, rule in sorted(zip(kinds, self.rules)))
        object.__setattr__(self, "rules", ordered)

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "rules": [rule.to_dict() for rule in self.rules],
        }
