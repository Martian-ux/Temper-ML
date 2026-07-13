"""Immutable run-attempt contract; lifecycle status remains event-derived."""

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
    require_non_negative_int,
    require_positive_int,
)


class EvaluationMode(str, Enum):
    """The user-selected quality-evaluation mode frozen for one run request."""

    NO_QUALITY_EVALUATION = "no_quality_evaluation"
    NONE = "no_quality_evaluation"
    LIGHT_EVALUATION = "light_evaluation"
    LIGHT = "light_evaluation"
    FULL_SUITE = "full_suite"
    FULL = "full_suite"
    EXPERIMENT_LOOP = "experiment_loop"


@dataclass(frozen=True)
class ResolvedRuntimeRequest(TypedRecord):
    """Immutable runtime inputs written before a worker can be launched."""

    RECORD_TYPE: ClassVar[str] = "resolved_runtime_request"

    request_id: str
    experiment: RecordReference
    experiment_manifest_identity: ContentIdentity
    recipe_resolution: RecordReference
    dataset_version_identity: ContentIdentity
    rendered_dataset_identity: ContentIdentity
    rendered_dataset_byte_count: int
    hardware_capability_profile: RecordReference
    execution_target: RecordReference
    runtime_identity: ContentIdentity
    preflight_identity: ContentIdentity
    training_state_identity: ContentIdentity
    evaluation_mode: EvaluationMode
    training_steps: int
    starting_step: int
    resume_from_run: RecordReference | None = None
    resume_checkpoint_identity: ContentIdentity | None = None

    def __post_init__(self) -> None:
        require_identifier("request_id", self.request_id)
        for field, record_type in (
            ("experiment", "experiment"),
            ("recipe_resolution", "recipe_resolution"),
            ("hardware_capability_profile", "hardware_capability_profile"),
            ("execution_target", "execution_target"),
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, RecordReference)
                or value.record_type != record_type
            ):
                raise RecordValidationError(f"{field} must reference {record_type}")
        for field in (
            "experiment_manifest_identity",
            "dataset_version_identity",
            "rendered_dataset_identity",
            "runtime_identity",
            "preflight_identity",
            "training_state_identity",
        ):
            if not isinstance(getattr(self, field), ContentIdentity):
                raise RecordValidationError(f"{field} must be a content identity")
        require_non_negative_int(
            "rendered_dataset_byte_count", self.rendered_dataset_byte_count
        )
        require_positive_int("training_steps", self.training_steps)
        require_non_negative_int("starting_step", self.starting_step)
        if self.starting_step >= self.training_steps:
            raise RecordValidationError("starting_step must precede training_steps")
        if not isinstance(self.evaluation_mode, EvaluationMode):
            raise RecordValidationError("evaluation_mode is invalid")
        if self.resume_from_run is not None and (
            not isinstance(self.resume_from_run, RecordReference)
            or self.resume_from_run.record_type != "run"
        ):
            raise RecordValidationError("resume_from_run must reference a run")
        if self.resume_checkpoint_identity is not None and not isinstance(
            self.resume_checkpoint_identity, ContentIdentity
        ):
            raise RecordValidationError(
                "resume_checkpoint_identity must be a content identity"
            )
        resuming = self.resume_from_run is not None
        if resuming != (self.resume_checkpoint_identity is not None):
            raise RecordValidationError(
                "resume run and checkpoint identity must be supplied together"
            )
        if resuming != (self.starting_step > 0):
            raise RecordValidationError(
                "resume evidence is required exactly when starting_step is non-zero"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "experiment": self.experiment.to_dict(),
            "experiment_manifest_identity": identity_fields(
                self.experiment_manifest_identity
            ),
            "recipe_resolution": self.recipe_resolution.to_dict(),
            "dataset_version_identity": identity_fields(self.dataset_version_identity),
            "rendered_dataset_identity": identity_fields(
                self.rendered_dataset_identity
            ),
            "rendered_dataset_byte_count": self.rendered_dataset_byte_count,
            "hardware_capability_profile": (self.hardware_capability_profile.to_dict()),
            "execution_target": self.execution_target.to_dict(),
            "runtime_identity": identity_fields(self.runtime_identity),
            "preflight_identity": identity_fields(self.preflight_identity),
            "training_state_identity": identity_fields(self.training_state_identity),
            "evaluation_mode": self.evaluation_mode.value,
            "training_steps": self.training_steps,
            "starting_step": self.starting_step,
            "resume_from_run": (
                self.resume_from_run.to_dict()
                if self.resume_from_run is not None
                else None
            ),
            "resume_checkpoint_identity": (
                identity_fields(self.resume_checkpoint_identity)
                if self.resume_checkpoint_identity is not None
                else None
            ),
        }


@dataclass(frozen=True)
class Run(TypedRecord):
    """One execution attempt of an unchanged experiment manifest."""

    RECORD_TYPE: ClassVar[str] = "run"

    run_id: str
    experiment: RecordReference
    experiment_manifest_identity: ContentIdentity
    attempt_number: int
    hardware_capability_profile: RecordReference
    execution_target: RecordReference
    runtime_identity: ContentIdentity
    request_identity: ContentIdentity
    training_state_identity: ContentIdentity
    retry_of: RecordReference | None = None

    def __post_init__(self) -> None:
        require_identifier("run_id", self.run_id)
        for field, record_type in (
            ("experiment", "experiment"),
            ("hardware_capability_profile", "hardware_capability_profile"),
            ("execution_target", "execution_target"),
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, RecordReference)
                or value.record_type != record_type
            ):
                raise RecordValidationError(f"{field} must reference {record_type}")
        require_positive_int("attempt_number", self.attempt_number)
        for field in (
            "experiment_manifest_identity",
            "runtime_identity",
            "request_identity",
            "training_state_identity",
        ):
            if not isinstance(getattr(self, field), ContentIdentity):
                raise RecordValidationError(f"{field} must be a content identity")
        if self.retry_of is not None and (
            not isinstance(self.retry_of, RecordReference)
            or self.retry_of.record_type != "run"
        ):
            raise RecordValidationError("retry_of must reference a run")
        if self.retry_of is not None and self.attempt_number == 1:
            raise RecordValidationError("the first attempt cannot be a retry")
        if self.retry_of is None and self.attempt_number != 1:
            raise RecordValidationError("later attempts must reference the prior run")

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment.to_dict(),
            "experiment_manifest_identity": identity_fields(
                self.experiment_manifest_identity
            ),
            "attempt_number": self.attempt_number,
            "hardware_capability_profile": self.hardware_capability_profile.to_dict(),
            "execution_target": self.execution_target.to_dict(),
            "runtime_identity": identity_fields(self.runtime_identity),
            "request_identity": identity_fields(self.request_identity),
            "training_state_identity": identity_fields(self.training_state_identity),
            "retry_of": self.retry_of.to_dict() if self.retry_of is not None else None,
        }
