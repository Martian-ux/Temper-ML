"""Project and immutable project-policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    identity_fields,
    require_identifier,
    require_string_tuple,
    require_text,
)


def _require_reference(
    field: str, value: RecordReference, record_type: str
) -> RecordReference:
    if not isinstance(value, RecordReference) or value.record_type != record_type:
        raise RecordValidationError(f"{field} must reference {record_type}")
    return value


def _require_identity(field: str, value: ContentIdentity) -> ContentIdentity:
    if not isinstance(value, ContentIdentity):
        raise RecordValidationError(f"{field} must be a content identity")
    return value


def _unique_references(
    field: str, values: tuple[RecordReference, ...], record_type: str
) -> tuple[RecordReference, ...]:
    if not isinstance(values, tuple):
        raise RecordValidationError(f"{field} must be a tuple")
    for value in values:
        _require_reference(field, value, record_type)
    keys = tuple(value.identity for value in values)
    if len(set(keys)) != len(keys):
        raise RecordValidationError(f"{field} must not contain duplicates")
    return tuple(
        sorted(values, key=lambda item: (item.identity.value, item.logical_id))
    )


@dataclass(frozen=True)
class Project(TypedRecord):
    """One immutable revision of a task-centered Temper project."""

    RECORD_TYPE: ClassVar[str] = "project"

    project_id: str
    display_name: str
    purpose: str
    task_definition: RecordReference
    base_model_revisions: tuple[RecordReference, ...] = ()

    def __post_init__(self) -> None:
        require_identifier("project_id", self.project_id)
        require_text("display_name", self.display_name)
        require_text("purpose", self.purpose)
        _require_reference("task_definition", self.task_definition, "task_definition")
        object.__setattr__(
            self,
            "base_model_revisions",
            _unique_references(
                "base_model_revisions",
                self.base_model_revisions,
                "base_model_revision",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "display_name": self.display_name,
            "purpose": self.purpose,
            "task_definition": self.task_definition.to_dict(),
            "base_model_revisions": [
                reference.to_dict() for reference in self.base_model_revisions
            ],
        }


@dataclass(frozen=True)
class ProjectPolicy(TypedRecord):
    """Immutable policy binding for one task-centered project revision."""

    RECORD_TYPE: ClassVar[str] = "project_policy"

    policy_id: str
    project: RecordReference
    task_definition: RecordReference
    rendering_contract: ContentIdentity
    evaluation_policy: ContentIdentity
    case_suites: tuple[ContentIdentity, ...]
    readiness_policy: ContentIdentity
    retention_policy: ContentIdentity
    approved_recipe_families: tuple[str, ...]
    baseline_policy: RecordReference
    recommendation_policy: ContentIdentity

    def __post_init__(self) -> None:
        require_identifier("policy_id", self.policy_id)
        _require_reference("project", self.project, "project")
        _require_reference("task_definition", self.task_definition, "task_definition")
        _require_reference("baseline_policy", self.baseline_policy, "baseline_policy")
        for field in (
            "rendering_contract",
            "evaluation_policy",
            "readiness_policy",
            "retention_policy",
            "recommendation_policy",
        ):
            _require_identity(field, getattr(self, field))
        if not isinstance(self.case_suites, tuple) or not self.case_suites:
            raise RecordValidationError("case_suites must be a non-empty tuple")
        for suite in self.case_suites:
            _require_identity("case_suites", suite)
        if len(set(self.case_suites)) != len(self.case_suites):
            raise RecordValidationError("case_suites must not contain duplicates")
        object.__setattr__(
            self, "case_suites", tuple(sorted(self.case_suites, key=str))
        )
        object.__setattr__(
            self,
            "approved_recipe_families",
            require_string_tuple(
                "approved_recipe_families",
                self.approved_recipe_families,
                sorted_values=True,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "project": self.project.to_dict(),
            "task_definition": self.task_definition.to_dict(),
            "rendering_contract": identity_fields(self.rendering_contract),
            "evaluation_policy": identity_fields(self.evaluation_policy),
            "case_suites": [identity_fields(suite) for suite in self.case_suites],
            "readiness_policy": identity_fields(self.readiness_policy),
            "retention_policy": identity_fields(self.retention_policy),
            "approved_recipe_families": list(self.approved_recipe_families),
            "baseline_policy": self.baseline_policy.to_dict(),
            "recommendation_policy": identity_fields(self.recommendation_policy),
        }
