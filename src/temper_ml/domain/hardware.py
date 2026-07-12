"""Public-safe hardware requirement, capability, and execution-target contracts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, ClassVar, Mapping

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    FrozenJsonObject,
    RecordReference,
    RecordValidationError,
    TypedRecord,
    freeze_json_object,
    identity_fields,
    require_identifier,
    require_non_negative_int,
    require_positive_int,
    require_string_tuple,
    require_text,
    thaw_json,
)

_PRIVATE_FACT_KEY = re.compile(
    r"(?:^|_)(?:host(?:name)?|user(?:name)?|serial|device_id|machine_id|"
    r"process_id|pid|ip|mac|path|home|account|organization|run_id)(?:_|$)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _public_facts(value: Mapping[str, Any], *, field: str) -> FrozenJsonObject:
    frozen = freeze_json_object(value, field=field)
    _reject_private_facts(thaw_json(frozen), field)
    return frozen


def _reject_private_facts(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _PRIVATE_FACT_KEY.search(key):
                raise RecordValidationError(f"{path} contains private capability key")
            _reject_private_facts(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_facts(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if (
            value.startswith(("/", "\\\\"))
            or _WINDOWS_ABSOLUTE.match(value)
            or "://" in value
        ):
            raise RecordValidationError(f"{path} contains a non-public location")


@dataclass(frozen=True)
class ExecutionTarget(TypedRecord):
    """One explicit execution-target class and its runtime contract."""

    RECORD_TYPE: ClassVar[str] = "execution_target"

    target_id: str
    target_class: str
    platform: str
    accelerator_backend: str
    runtime_contract: ContentIdentity
    capabilities: tuple[str, ...]
    constraints: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_identifier("target_id", self.target_id)
        require_identifier("target_class", self.target_class)
        require_identifier("platform", self.platform)
        require_identifier("accelerator_backend", self.accelerator_backend)
        if not isinstance(self.runtime_contract, ContentIdentity):
            raise RecordValidationError("runtime_contract must be a content identity")
        object.__setattr__(
            self,
            "capabilities",
            require_string_tuple(
                "capabilities", self.capabilities, non_empty=False, sorted_values=True
            ),
        )
        object.__setattr__(
            self,
            "constraints",
            _public_facts(self.constraints, field="constraints"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "target_class": self.target_class,
            "platform": self.platform,
            "accelerator_backend": self.accelerator_backend,
            "runtime_contract": identity_fields(self.runtime_contract),
            "capabilities": list(self.capabilities),
            "constraints": thaw_json(self.constraints),
        }


@dataclass(frozen=True)
class HardwareRequirements(TypedRecord):
    """Portable machine constraints bound to an immutable experiment."""

    RECORD_TYPE: ClassVar[str] = "hardware_requirements"

    requirements_id: str
    execution_target_classes: tuple[str, ...]
    accelerator_backends: tuple[str, ...]
    minimum_accelerator_memory_bytes: int
    minimum_system_memory_bytes: int
    required_precision_modes: tuple[str, ...]
    required_quantization_modes: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    constraints: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_identifier("requirements_id", self.requirements_id)
        for field in ("execution_target_classes", "accelerator_backends"):
            object.__setattr__(
                self,
                field,
                require_string_tuple(field, getattr(self, field), sorted_values=True),
            )
        require_non_negative_int(
            "minimum_accelerator_memory_bytes",
            self.minimum_accelerator_memory_bytes,
        )
        require_non_negative_int(
            "minimum_system_memory_bytes", self.minimum_system_memory_bytes
        )
        for field in (
            "required_precision_modes",
            "required_quantization_modes",
            "required_capabilities",
        ):
            object.__setattr__(
                self,
                field,
                require_string_tuple(
                    field,
                    getattr(self, field),
                    non_empty=False,
                    sorted_values=True,
                ),
            )
        object.__setattr__(
            self,
            "constraints",
            _public_facts(self.constraints, field="constraints"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "requirements_id": self.requirements_id,
            "execution_target_classes": list(self.execution_target_classes),
            "accelerator_backends": list(self.accelerator_backends),
            "minimum_accelerator_memory_bytes": self.minimum_accelerator_memory_bytes,
            "minimum_system_memory_bytes": self.minimum_system_memory_bytes,
            "required_precision_modes": list(self.required_precision_modes),
            "required_quantization_modes": list(self.required_quantization_modes),
            "required_capabilities": list(self.required_capabilities),
            "constraints": thaw_json(self.constraints),
        }


@dataclass(frozen=True)
class HardwareCapabilityProfile(TypedRecord):
    """Sanitized observed capability facts attached to one run attempt."""

    RECORD_TYPE: ClassVar[str] = "hardware_capability_profile"

    profile_id: str
    execution_target: RecordReference
    accelerator_backend: str
    accelerator_architecture: str
    accelerator_model: str
    accelerator_count: int
    accelerator_memory_bytes: tuple[int, ...]
    system_memory_bytes: int
    supported_precision_modes: tuple[str, ...]
    supported_quantization_modes: tuple[str, ...]
    capabilities: tuple[str, ...]
    library_versions: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_identifier("profile_id", self.profile_id)
        if (
            not isinstance(self.execution_target, RecordReference)
            or self.execution_target.record_type != "execution_target"
        ):
            raise RecordValidationError(
                "execution_target must reference an execution_target"
            )
        require_identifier("accelerator_backend", self.accelerator_backend)
        require_text("accelerator_architecture", self.accelerator_architecture)
        require_text("accelerator_model", self.accelerator_model)
        require_non_negative_int("accelerator_count", self.accelerator_count)
        if (
            not isinstance(self.accelerator_memory_bytes, tuple)
            or len(self.accelerator_memory_bytes) != self.accelerator_count
        ):
            raise RecordValidationError(
                "accelerator_memory_bytes must contain one value per accelerator"
            )
        for value in self.accelerator_memory_bytes:
            require_positive_int("accelerator_memory_bytes", value)
        require_positive_int("system_memory_bytes", self.system_memory_bytes)
        for field in (
            "supported_precision_modes",
            "supported_quantization_modes",
            "capabilities",
        ):
            object.__setattr__(
                self,
                field,
                require_string_tuple(
                    field,
                    getattr(self, field),
                    non_empty=False,
                    sorted_values=True,
                ),
            )
        object.__setattr__(
            self,
            "library_versions",
            _public_facts(self.library_versions, field="library_versions"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "execution_target": self.execution_target.to_dict(),
            "accelerator_backend": self.accelerator_backend,
            "accelerator_architecture": self.accelerator_architecture,
            "accelerator_model": self.accelerator_model,
            "accelerator_count": self.accelerator_count,
            "accelerator_memory_bytes": list(self.accelerator_memory_bytes),
            "system_memory_bytes": self.system_memory_bytes,
            "supported_precision_modes": list(self.supported_precision_modes),
            "supported_quantization_modes": list(self.supported_quantization_modes),
            "capabilities": list(self.capabilities),
            "library_versions": thaw_json(self.library_versions),
        }
