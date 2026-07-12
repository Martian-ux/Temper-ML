"""Immutable run-attempt contract; lifecycle status remains event-derived."""

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
    require_positive_int,
)


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
