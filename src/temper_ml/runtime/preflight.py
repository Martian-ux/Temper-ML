"""Hardware capture, target selection, estimates, and deterministic preflight."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import record_reference, thaw_json


class PreflightError(RuntimeError):
    """A stable hardware-selection or preflight failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _non_negative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PreflightError(f"{name}_invalid")
    return value


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PreflightError(f"{name}_invalid")
    return value


@dataclass(frozen=True)
class EstimateComponents:
    """Explicit byte contributions used by the transparent estimate formula."""

    base_model_bytes: int
    adapter_optimizer_bytes: int
    peak_activation_bytes: int
    accelerator_runtime_overhead_bytes: int
    dataset_bytes: int
    host_runtime_overhead_bytes: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _non_negative_int(field, getattr(self, field))


@dataclass(frozen=True)
class PreflightEstimate:
    """A deterministic estimate with no hidden multipliers."""

    accelerator_memory_bytes: int
    system_memory_bytes: int
    training_steps: int
    effective_batch_size: int

    def __post_init__(self) -> None:
        _validate_preflight_estimate(self)

    def to_dict(self) -> dict[str, int]:
        return {
            "accelerator_memory_bytes": self.accelerator_memory_bytes,
            "system_memory_bytes": self.system_memory_bytes,
            "training_steps": self.training_steps,
            "effective_batch_size": self.effective_batch_size,
        }


def _validate_preflight_estimate(estimate: PreflightEstimate) -> None:
    _non_negative_int("accelerator_memory_bytes", estimate.accelerator_memory_bytes)
    _non_negative_int("system_memory_bytes", estimate.system_memory_bytes)
    _positive_int("training_steps", estimate.training_steps)
    _positive_int("effective_batch_size", estimate.effective_batch_size)


def _validate_estimate_resolution(
    estimate: PreflightEstimate, resolution: RecipeResolution
) -> None:
    if estimate.training_steps != resolution.training_steps:
        raise PreflightError("estimate_training_steps_mismatch")
    if estimate.effective_batch_size != resolution.effective_batch_size:
        raise PreflightError("estimate_effective_batch_size_mismatch")


def estimate_resources(
    resolution: RecipeResolution,
    components: EstimateComponents,
) -> PreflightEstimate:
    """Add explicit byte components and bind the result to resolved training size."""

    if not isinstance(resolution, RecipeResolution):
        raise PreflightError("recipe_resolution_invalid")
    if not isinstance(components, EstimateComponents):
        raise PreflightError("estimate_components_invalid")
    accelerator = (
        components.base_model_bytes
        + components.adapter_optimizer_bytes
        + components.peak_activation_bytes
        + components.accelerator_runtime_overhead_bytes
    )
    system = (
        components.base_model_bytes
        + components.dataset_bytes
        + components.host_runtime_overhead_bytes
    )
    return PreflightEstimate(
        accelerator,
        system,
        resolution.training_steps,
        resolution.effective_batch_size,
    )


class PreflightStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ConstraintCheck:
    """One inspectable, machine-readable preflight constraint decision."""

    code: str
    satisfied: bool
    required: object
    observed: object

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "satisfied": self.satisfied,
            "required": _view_value(self.required),
            "observed": _view_value(self.observed),
        }


@dataclass(frozen=True)
class PreflightResult:
    """The exact target, profile, estimate, and every constraint decision."""

    resolution: RecipeResolution
    requirements: HardwareRequirements
    target: ExecutionTarget
    profile: HardwareCapabilityProfile
    estimate: PreflightEstimate
    checks: tuple[ConstraintCheck, ...]

    def __post_init__(self) -> None:
        expected = _canonical_constraint_checks(
            self.resolution,
            self.requirements,
            self.target,
            self.profile,
            self.estimate,
        )
        if not isinstance(self.checks, tuple) or any(
            not isinstance(check, ConstraintCheck) for check in self.checks
        ):
            raise PreflightError("preflight_checks_invalid")
        if self.checks != expected:
            raise PreflightError("preflight_checks_mismatch")

    @property
    def status(self) -> PreflightStatus:
        return (
            PreflightStatus.READY
            if all(check.satisfied for check in self.checks)
            else PreflightStatus.BLOCKED
        )

    @property
    def ready(self) -> bool:
        return self.status is PreflightStatus.READY

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        return tuple(check.code for check in self.checks if not check.satisfied)

    def to_view(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "resolution": record_reference(self.resolution).to_dict(),
            "hardware_requirements": record_reference(self.requirements).to_dict(),
            "execution_target": record_reference(self.target).to_dict(),
            "hardware_capability_profile": record_reference(self.profile).to_dict(),
            "estimate": self.estimate.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "blocking_reasons": list(self.blocking_reasons),
        }


def capture_capability_profile(
    *,
    profile_id: str,
    execution_target: ExecutionTarget,
    accelerator_backend: str,
    accelerator_architecture: str,
    accelerator_model: str,
    accelerator_count: int,
    accelerator_memory_bytes: tuple[int, ...],
    system_memory_bytes: int,
    supported_precision_modes: tuple[str, ...],
    supported_quantization_modes: tuple[str, ...],
    capabilities: tuple[str, ...],
    library_versions: Mapping[str, Any],
) -> HardwareCapabilityProfile:
    """Capture only the sanitized facts admitted by the Slice 2 contract."""

    if not isinstance(execution_target, ExecutionTarget):
        raise PreflightError("execution_target_invalid")
    if accelerator_backend != execution_target.accelerator_backend:
        raise PreflightError("capability_backend_mismatch")
    try:
        return HardwareCapabilityProfile(
            profile_id=profile_id,
            execution_target=record_reference(execution_target),
            accelerator_backend=accelerator_backend,
            accelerator_architecture=accelerator_architecture,
            accelerator_model=accelerator_model,
            accelerator_count=accelerator_count,
            accelerator_memory_bytes=accelerator_memory_bytes,
            system_memory_bytes=system_memory_bytes,
            supported_precision_modes=supported_precision_modes,
            supported_quantization_modes=supported_quantization_modes,
            capabilities=capabilities,
            library_versions=library_versions,
        )
    except (TypeError, ValueError):
        raise PreflightError("capability_profile_invalid") from None


def select_execution_target(
    requirements: HardwareRequirements,
    targets: Iterable[ExecutionTarget],
    *,
    selected_target_id: str | None = None,
) -> ExecutionTarget:
    """Select explicitly when more than one compatible target is available."""

    if not isinstance(requirements, HardwareRequirements):
        raise PreflightError("hardware_requirements_invalid")
    values = tuple(targets)
    if any(not isinstance(target, ExecutionTarget) for target in values):
        raise PreflightError("execution_target_invalid")
    compatible = tuple(
        sorted(
            (target for target in values if _target_supported(requirements, target)),
            key=lambda target: target.target_id,
        )
    )
    if selected_target_id is not None:
        selected = tuple(
            target for target in values if target.target_id == selected_target_id
        )
        if len(selected) != 1:
            raise PreflightError("selected_target_not_found")
        if selected[0] not in compatible:
            raise PreflightError("selected_target_incompatible")
        return selected[0]
    if not compatible:
        raise PreflightError("execution_target_unavailable")
    if len(compatible) != 1:
        raise PreflightError("execution_target_selection_required")
    return compatible[0]


def preflight(
    resolution: RecipeResolution,
    requirements: HardwareRequirements,
    target: ExecutionTarget,
    profile: HardwareCapabilityProfile,
    estimate: PreflightEstimate,
) -> PreflightResult:
    """Evaluate one already-selected target without substituting another target."""

    checks = _canonical_constraint_checks(
        resolution,
        requirements,
        target,
        profile,
        estimate,
    )
    return PreflightResult(
        resolution,
        requirements,
        target,
        profile,
        estimate,
        checks,
    )


def _canonical_constraint_checks(
    resolution: RecipeResolution,
    requirements: HardwareRequirements,
    target: ExecutionTarget,
    profile: HardwareCapabilityProfile,
    estimate: PreflightEstimate,
) -> tuple[ConstraintCheck, ...]:
    """Build the one complete, deterministic set of preflight decisions."""

    if not isinstance(resolution, RecipeResolution):
        raise PreflightError("recipe_resolution_invalid")
    if not isinstance(requirements, HardwareRequirements):
        raise PreflightError("hardware_requirements_invalid")
    if not isinstance(target, ExecutionTarget):
        raise PreflightError("execution_target_invalid")
    if not isinstance(profile, HardwareCapabilityProfile):
        raise PreflightError("capability_profile_invalid")
    if not isinstance(estimate, PreflightEstimate):
        raise PreflightError("preflight_estimate_invalid")
    _validate_preflight_estimate(estimate)
    _validate_estimate_resolution(estimate, resolution)
    if resolution.hardware_requirements != record_reference(requirements):
        raise PreflightError("resolution_requirements_mismatch")
    if resolution.execution_target != record_reference(target):
        raise PreflightError("resolution_target_mismatch")
    if profile.execution_target != record_reference(target):
        raise PreflightError("profile_target_mismatch")

    available_accelerator_memory = (
        max(profile.accelerator_memory_bytes) if profile.accelerator_memory_bytes else 0
    )
    required_capabilities = tuple(sorted(requirements.required_capabilities))
    required_precision = tuple(sorted(requirements.required_precision_modes))
    required_quantization = tuple(sorted(requirements.required_quantization_modes))
    target_capabilities = tuple(sorted(target.capabilities))
    profile_capabilities = tuple(sorted(profile.capabilities))
    required_library_versions = thaw_json(resolution.library_versions)
    observed_library_versions = thaw_json(profile.library_versions)
    if not isinstance(required_library_versions, dict) or not isinstance(
        observed_library_versions, dict
    ):
        raise PreflightError("library_versions_invalid")
    library_versions_match = all(
        name in observed_library_versions
        and observed_library_versions[name] == expected
        for name, expected in required_library_versions.items()
    )
    return (
        ConstraintCheck(
            "target_class_allowed",
            target.target_class in requirements.execution_target_classes,
            requirements.execution_target_classes,
            target.target_class,
        ),
        ConstraintCheck(
            "accelerator_backend_allowed",
            target.accelerator_backend in requirements.accelerator_backends,
            requirements.accelerator_backends,
            target.accelerator_backend,
        ),
        ConstraintCheck(
            "profile_backend_matches",
            profile.accelerator_backend == target.accelerator_backend,
            target.accelerator_backend,
            profile.accelerator_backend,
        ),
        ConstraintCheck(
            "target_capabilities_present",
            not (set(required_capabilities) - set(target_capabilities)),
            required_capabilities,
            target_capabilities,
        ),
        ConstraintCheck(
            "profile_capabilities_present",
            not (set(required_capabilities) - set(profile_capabilities)),
            required_capabilities,
            profile_capabilities,
        ),
        ConstraintCheck(
            "resolved_library_versions_match",
            library_versions_match,
            required_library_versions,
            observed_library_versions,
        ),
        ConstraintCheck(
            "minimum_accelerator_memory",
            available_accelerator_memory
            >= requirements.minimum_accelerator_memory_bytes,
            requirements.minimum_accelerator_memory_bytes,
            available_accelerator_memory,
        ),
        ConstraintCheck(
            "estimated_accelerator_memory",
            estimate.accelerator_memory_bytes <= available_accelerator_memory,
            estimate.accelerator_memory_bytes,
            available_accelerator_memory,
        ),
        ConstraintCheck(
            "minimum_system_memory",
            profile.system_memory_bytes >= requirements.minimum_system_memory_bytes,
            requirements.minimum_system_memory_bytes,
            profile.system_memory_bytes,
        ),
        ConstraintCheck(
            "estimated_system_memory",
            estimate.system_memory_bytes <= profile.system_memory_bytes,
            estimate.system_memory_bytes,
            profile.system_memory_bytes,
        ),
        ConstraintCheck(
            "required_precision_modes",
            not (set(required_precision) - set(profile.supported_precision_modes)),
            required_precision,
            profile.supported_precision_modes,
        ),
        ConstraintCheck(
            "resolved_precision_supported",
            resolution.precision in profile.supported_precision_modes,
            resolution.precision,
            profile.supported_precision_modes,
        ),
        ConstraintCheck(
            "required_quantization_modes",
            not (
                set(required_quantization) - set(profile.supported_quantization_modes)
            ),
            required_quantization,
            profile.supported_quantization_modes,
        ),
        ConstraintCheck(
            "resolved_quantization_supported",
            resolution.quantization == "none"
            or resolution.quantization in profile.supported_quantization_modes,
            resolution.quantization,
            profile.supported_quantization_modes,
        ),
    )


def material_change_reasons(
    original_target: ExecutionTarget,
    candidate_target: ExecutionTarget,
    original_resolution: RecipeResolution,
    candidate_resolution: RecipeResolution,
) -> tuple[str, ...]:
    """Report only experiment-material target or resolved-manifest changes."""

    for value, code in (
        (original_target, "execution_target_invalid"),
        (candidate_target, "execution_target_invalid"),
    ):
        if not isinstance(value, ExecutionTarget):
            raise PreflightError(code)
    for resolution_value in (original_resolution, candidate_resolution):
        if not isinstance(resolution_value, RecipeResolution):
            raise PreflightError("recipe_resolution_invalid")
    reasons: list[str] = []
    if original_target.target_class != candidate_target.target_class:
        reasons.append("execution_target_class_changed")
    if original_target.accelerator_backend != candidate_target.accelerator_backend:
        reasons.append("accelerator_backend_changed")
    if original_target.identity != candidate_target.identity:
        reasons.append("execution_target_revision_changed")
    original_payload = original_resolution.to_payload()
    candidate_payload = candidate_resolution.to_payload()
    original_payload.pop("resolution_id")
    candidate_payload.pop("resolution_id")
    if original_payload != candidate_payload:
        reasons.append("recipe_resolution_changed")
    if original_resolution.library_versions != candidate_resolution.library_versions:
        reasons.append("runtime_library_versions_changed")
    technical_fields = set(original_payload) - {
        "recipe",
        "base_model_revision",
        "hardware_requirements",
        "execution_target",
        "library_versions",
        "applied_constraints",
    }
    if any(
        original_payload[field] != candidate_payload[field]
        for field in technical_fields
    ):
        reasons.append("training_configuration_changed")
    return tuple(reasons)


def _target_supported(
    requirements: HardwareRequirements, target: ExecutionTarget
) -> bool:
    return (
        target.target_class in requirements.execution_target_classes
        and target.accelerator_backend in requirements.accelerator_backends
        and not (set(requirements.required_capabilities) - set(target.capabilities))
    )


def _view_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_view_value(item) for item in value]
    return value
