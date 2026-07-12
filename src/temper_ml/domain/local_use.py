"""Focused local adapter-use and verified export record contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from temper_ml.domain.artifacts import StorageReference
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    FrozenJsonObject,
    RecordReference,
    RecordValidationError,
    TypedRecord,
    freeze_json_object,
    identity_fields,
    require_identifier,
    thaw_json,
)


def _reference(field: str, value: RecordReference, kind: str) -> RecordReference:
    if not isinstance(value, RecordReference) or value.record_type != kind:
        raise RecordValidationError(f"{field} must reference {kind}")
    return value


def _identity(field: str, value: ContentIdentity) -> ContentIdentity:
    if not isinstance(value, ContentIdentity):
        raise RecordValidationError(f"{field} must be a content identity")
    return value


def _frozen_items(
    field: str, values: tuple[Mapping[str, Any], ...]
) -> tuple[FrozenJsonObject, ...]:
    if not isinstance(values, tuple) or not values:
        raise RecordValidationError(f"{field} must be a non-empty tuple")
    return tuple(
        freeze_json_object(value, field=f"{field}[{index}]")
        for index, value in enumerate(values)
    )


@dataclass(frozen=True)
class LocalUseSession(TypedRecord):
    """A saved, focused use session pinned to verified runtime inputs."""

    RECORD_TYPE: ClassVar[str] = "local_use_session"

    session_id: str
    project: RecordReference
    artifact: RecordReference
    artifact_content_identity: ContentIdentity
    base_model_revision: RecordReference
    tokenizer_identity: ContentIdentity
    compatibility_group: RecordReference
    execution_target: RecordReference
    inference_settings: Mapping[str, Any]
    inputs: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    runtime_evidence: ContentIdentity
    integrity_evidence: ContentIdentity
    evaluation_captures: tuple[ContentIdentity, ...] = ()

    def __post_init__(self) -> None:
        require_identifier("session_id", self.session_id)
        for field, kind in (
            ("project", "project"),
            ("artifact", "artifact"),
            ("base_model_revision", "base_model_revision"),
            ("compatibility_group", "compatibility_group"),
            ("execution_target", "execution_target"),
        ):
            _reference(field, getattr(self, field), kind)
        for field in (
            "artifact_content_identity",
            "tokenizer_identity",
            "runtime_evidence",
            "integrity_evidence",
        ):
            _identity(field, getattr(self, field))
        object.__setattr__(
            self,
            "inference_settings",
            freeze_json_object(self.inference_settings, field="inference_settings"),
        )
        object.__setattr__(self, "inputs", _frozen_items("inputs", self.inputs))
        object.__setattr__(self, "outputs", _frozen_items("outputs", self.outputs))
        if len(self.inputs) != len(self.outputs):
            raise RecordValidationError("inputs and outputs must have equal length")
        if not isinstance(self.evaluation_captures, tuple):
            raise RecordValidationError("evaluation_captures must be a tuple")
        for capture in self.evaluation_captures:
            _identity("evaluation_captures", capture)
        if len(set(self.evaluation_captures)) != len(self.evaluation_captures):
            raise RecordValidationError(
                "evaluation_captures must not contain duplicates"
            )
        object.__setattr__(
            self,
            "evaluation_captures",
            tuple(sorted(self.evaluation_captures, key=str)),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "project": self.project.to_dict(),
            "artifact": self.artifact.to_dict(),
            "artifact_content_identity": identity_fields(
                self.artifact_content_identity
            ),
            "base_model_revision": self.base_model_revision.to_dict(),
            "tokenizer_identity": identity_fields(self.tokenizer_identity),
            "compatibility_group": self.compatibility_group.to_dict(),
            "execution_target": self.execution_target.to_dict(),
            "inference_settings": thaw_json(self.inference_settings),
            "inputs": [thaw_json(value) for value in self.inputs],
            "outputs": [thaw_json(value) for value in self.outputs],
            "runtime_evidence": identity_fields(self.runtime_evidence),
            "integrity_evidence": identity_fields(self.integrity_evidence),
            "evaluation_captures": [
                identity_fields(capture) for capture in self.evaluation_captures
            ],
        }


@dataclass(frozen=True)
class AdapterExport(TypedRecord):
    """Portable adapter export evidence without deployment/readiness semantics."""

    RECORD_TYPE: ClassVar[str] = "adapter_export"

    export_id: str
    artifact: RecordReference
    adapter_content_identity: ContentIdentity
    integrity_manifest_identity: ContentIdentity
    integrity_evidence: ContentIdentity
    compatibility_groups: tuple[RecordReference, ...]
    compatibility_requirements: Mapping[str, Any]
    provenance: ContentIdentity
    export_format: str
    storage_reference: StorageReference

    def __post_init__(self) -> None:
        require_identifier("export_id", self.export_id)
        _reference("artifact", self.artifact, "artifact")
        for field in (
            "adapter_content_identity",
            "integrity_manifest_identity",
            "integrity_evidence",
            "provenance",
        ):
            _identity(field, getattr(self, field))
        if (
            not isinstance(self.compatibility_groups, tuple)
            or not self.compatibility_groups
        ):
            raise RecordValidationError(
                "compatibility_groups must be a non-empty tuple"
            )
        for group in self.compatibility_groups:
            _reference("compatibility_groups", group, "compatibility_group")
        keys = tuple(group.identity for group in self.compatibility_groups)
        if len(set(keys)) != len(keys):
            raise RecordValidationError(
                "compatibility_groups must not contain duplicates"
            )
        object.__setattr__(
            self,
            "compatibility_groups",
            tuple(
                sorted(
                    self.compatibility_groups,
                    key=lambda group: (group.identity.value, group.logical_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "compatibility_requirements",
            freeze_json_object(
                self.compatibility_requirements,
                field="compatibility_requirements",
            ),
        )
        require_identifier("export_format", self.export_format)
        if not isinstance(self.storage_reference, StorageReference):
            raise RecordValidationError(
                "storage_reference must be a logical StorageReference"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "export_id": self.export_id,
            "artifact": self.artifact.to_dict(),
            "adapter_content_identity": identity_fields(self.adapter_content_identity),
            "integrity_manifest_identity": identity_fields(
                self.integrity_manifest_identity
            ),
            "integrity_evidence": identity_fields(self.integrity_evidence),
            "compatibility_groups": [
                group.to_dict() for group in self.compatibility_groups
            ],
            "compatibility_requirements": thaw_json(self.compatibility_requirements),
            "provenance": identity_fields(self.provenance),
            "export_format": self.export_format,
            "storage_reference": self.storage_reference.to_dict(),
        }
