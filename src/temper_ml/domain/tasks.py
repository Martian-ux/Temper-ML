"""Task-definition domain contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    FrozenJsonObject,
    RecordValidationError,
    TypedRecord,
    freeze_json_object,
    identity_fields,
    require_identifier,
    require_string_tuple,
    require_text,
    thaw_json,
)


@dataclass(frozen=True)
class TaskDefinition(TypedRecord):
    """Immutable task inputs, outputs, rendering, and comparison objectives."""

    RECORD_TYPE: ClassVar[str] = "task_definition"

    task_id: str
    display_name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    rendering_contract: ContentIdentity
    objectives: tuple[str, ...]
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier("task_id", self.task_id)
        require_text("display_name", self.display_name)
        require_text("description", self.description)
        if not isinstance(self.rendering_contract, ContentIdentity):
            raise RecordValidationError("rendering_contract must be a content identity")
        object.__setattr__(
            self,
            "input_schema",
            freeze_json_object(self.input_schema, field="input_schema"),
        )
        object.__setattr__(
            self,
            "output_schema",
            freeze_json_object(self.output_schema, field="output_schema"),
        )
        object.__setattr__(
            self,
            "objectives",
            require_string_tuple("objectives", self.objectives),
        )
        object.__setattr__(
            self,
            "capabilities",
            require_string_tuple(
                "capabilities", self.capabilities, non_empty=False, sorted_values=True
            ),
        )

    def to_payload(self) -> dict[str, object]:
        input_schema = self.input_schema
        output_schema = self.output_schema
        if not isinstance(input_schema, FrozenJsonObject) or not isinstance(
            output_schema, FrozenJsonObject
        ):
            raise RecordValidationError("task schemas are not frozen")
        return {
            "task_id": self.task_id,
            "display_name": self.display_name,
            "description": self.description,
            "input_schema": thaw_json(input_schema),
            "output_schema": thaw_json(output_schema),
            "rendering_contract": identity_fields(self.rendering_contract),
            "objectives": list(self.objectives),
            "capabilities": list(self.capabilities),
        }
