"""Explicit compatibility groups and relationship validators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from temper_ml.domain.artifacts import Artifact
from temper_ml.domain.hardware import ExecutionTarget
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    identity_fields,
    require_identifier,
    require_string_tuple,
)


class CompatibilityRelationship(str, Enum):
    COMPARABLE = "comparable"
    MERGE = "merge_compatible"
    RESUME = "resume_compatible"
    DEPLOYMENT = "deployment_compatible"


class CompatibilityError(ValueError):
    """Raised when a required compatibility relationship does not hold."""

    def __init__(self, decision: "CompatibilityDecision") -> None:
        self.decision = decision
        codes = ", ".join(violation.code for violation in decision.violations)
        super().__init__(f"{decision.relationship.value} rejected: {codes}")


@dataclass(frozen=True)
class CompatibilityViolation:
    """One deterministic, machine-readable incompatibility reason."""

    code: str
    field: str

    def __post_init__(self) -> None:
        require_identifier("compatibility violation code", self.code)
        require_identifier("compatibility violation field", self.field)

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field}


@dataclass(frozen=True)
class CompatibilityDecision:
    """A relationship decision that never relies on display names."""

    relationship: CompatibilityRelationship
    violations: tuple[CompatibilityViolation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.relationship, CompatibilityRelationship):
            raise RecordValidationError("compatibility relationship is invalid")
        if not isinstance(self.violations, tuple) or any(
            not isinstance(item, CompatibilityViolation) for item in self.violations
        ):
            raise RecordValidationError("compatibility violations are invalid")
        codes = tuple((item.code, item.field) for item in self.violations)
        if len(set(codes)) != len(codes):
            raise RecordValidationError("compatibility violations must be unique")

    @property
    def compatible(self) -> bool:
        return not self.violations

    def require(self) -> None:
        if not self.compatible:
            raise CompatibilityError(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "relationship": self.relationship.value,
            "compatible": self.compatible,
            "violations": [item.to_dict() for item in self.violations],
        }


@dataclass(frozen=True)
class RuntimeTargetConstraint:
    """Structural requirements for one explicit execution-target class."""

    target_class: str
    accelerator_backend: str
    runtime_contract: ContentIdentity
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier("target_class", self.target_class)
        require_identifier("accelerator_backend", self.accelerator_backend)
        if not isinstance(self.runtime_contract, ContentIdentity):
            raise RecordValidationError("runtime_contract must be a content identity")
        object.__setattr__(
            self,
            "required_capabilities",
            require_string_tuple(
                "required_capabilities",
                self.required_capabilities,
                non_empty=False,
                sorted_values=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_class": self.target_class,
            "accelerator_backend": self.accelerator_backend,
            "runtime_contract": identity_fields(self.runtime_contract),
            "required_capabilities": list(self.required_capabilities),
        }


@dataclass(frozen=True)
class CompatibilityGroup(TypedRecord):
    """Exact model, tokenizer, adapter, module, and runtime constraints."""

    RECORD_TYPE: ClassVar[str] = "compatibility_group"

    group_id: str
    base_model_revision: RecordReference
    tokenizer_identity: ContentIdentity
    rendering_template: ContentIdentity
    adapter_type: str
    target_modules: tuple[str, ...]
    runtime_targets: tuple[RuntimeTargetConstraint, ...]
    merge_methods: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier("group_id", self.group_id)
        if (
            not isinstance(self.base_model_revision, RecordReference)
            or self.base_model_revision.record_type != "base_model_revision"
        ):
            raise RecordValidationError(
                "base_model_revision must reference a base_model_revision"
            )
        for field in ("tokenizer_identity", "rendering_template"):
            if not isinstance(getattr(self, field), ContentIdentity):
                raise RecordValidationError(f"{field} must be a content identity")
        require_identifier("adapter_type", self.adapter_type)
        object.__setattr__(
            self,
            "target_modules",
            require_string_tuple(
                "target_modules", self.target_modules, sorted_values=True
            ),
        )
        if not isinstance(self.runtime_targets, tuple) or not self.runtime_targets:
            raise RecordValidationError("runtime_targets must be a non-empty tuple")
        if any(
            not isinstance(target, RuntimeTargetConstraint)
            for target in self.runtime_targets
        ):
            raise RecordValidationError("runtime_targets contains an invalid target")
        classes = tuple(target.target_class for target in self.runtime_targets)
        if len(set(classes)) != len(classes):
            raise RecordValidationError("runtime target classes must be unique")
        object.__setattr__(
            self,
            "runtime_targets",
            tuple(sorted(self.runtime_targets, key=lambda target: target.target_class)),
        )
        object.__setattr__(
            self,
            "merge_methods",
            require_string_tuple(
                "merge_methods", self.merge_methods, non_empty=False, sorted_values=True
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "base_model_revision": self.base_model_revision.to_dict(),
            "tokenizer_identity": identity_fields(self.tokenizer_identity),
            "rendering_template": identity_fields(self.rendering_template),
            "adapter_type": self.adapter_type,
            "target_modules": list(self.target_modules),
            "runtime_targets": [target.to_dict() for target in self.runtime_targets],
            "merge_methods": list(self.merge_methods),
        }

    def target_constraint(self, target_class: str) -> RuntimeTargetConstraint | None:
        return next(
            (
                target
                for target in self.runtime_targets
                if target.target_class == target_class
            ),
            None,
        )


@dataclass(frozen=True)
class ComparisonProfile:
    """Only the policy facts relevant to cross-candidate comparison."""

    task_definition: RecordReference
    project_policy: RecordReference
    evaluation_policy: ContentIdentity
    objectives: tuple[str, ...]
    base_model_revision: RecordReference

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_definition, RecordReference)
            or self.task_definition.record_type != "task_definition"
        ):
            raise RecordValidationError(
                "task_definition must reference a task_definition"
            )
        if (
            not isinstance(self.project_policy, RecordReference)
            or self.project_policy.record_type != "project_policy"
        ):
            raise RecordValidationError(
                "project_policy must reference a project_policy"
            )
        if not isinstance(self.evaluation_policy, ContentIdentity):
            raise RecordValidationError("evaluation_policy must be a content identity")
        object.__setattr__(
            self,
            "objectives",
            require_string_tuple("objectives", self.objectives),
        )
        if (
            not isinstance(self.base_model_revision, RecordReference)
            or self.base_model_revision.record_type != "base_model_revision"
        ):
            raise RecordValidationError(
                "base_model_revision must reference a base_model_revision"
            )


@dataclass(frozen=True)
class ResumeCheckpoint:
    """Retained checkpoint facts relevant to an exact resume decision."""

    experiment_manifest_identity: ContentIdentity
    recipe_resolution: RecordReference
    training_state_identity: ContentIdentity
    execution_target: RecordReference
    available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_manifest_identity, ContentIdentity):
            raise RecordValidationError(
                "experiment_manifest_identity must be a content identity"
            )
        if (
            not isinstance(self.recipe_resolution, RecordReference)
            or self.recipe_resolution.record_type != "recipe_resolution"
        ):
            raise RecordValidationError(
                "recipe_resolution must reference a recipe_resolution"
            )
        if not isinstance(self.training_state_identity, ContentIdentity):
            raise RecordValidationError(
                "training_state_identity must be a content identity"
            )
        if (
            not isinstance(self.execution_target, RecordReference)
            or self.execution_target.record_type != "execution_target"
        ):
            raise RecordValidationError(
                "execution_target must reference an execution_target"
            )
        if not isinstance(self.available, bool):
            raise RecordValidationError("available must be a boolean")


@dataclass(frozen=True)
class ResumeRequest:
    """Requested immutable training facts for a resume attempt."""

    experiment_manifest_identity: ContentIdentity
    recipe_resolution: RecordReference
    training_state_identity: ContentIdentity
    execution_target: RecordReference

    def __post_init__(self) -> None:
        ResumeCheckpoint(
            self.experiment_manifest_identity,
            self.recipe_resolution,
            self.training_state_identity,
            self.execution_target,
            True,
        )


def check_comparison_compatibility(
    left: ComparisonProfile, right: ComparisonProfile
) -> CompatibilityDecision:
    """Check shared task policy/objectives while allowing different models."""

    violations: list[CompatibilityViolation] = []
    if left.project_policy.identity != right.project_policy.identity:
        violations.append(
            CompatibilityViolation("project_policy_mismatch", "project_policy")
        )
    if left.task_definition.identity != right.task_definition.identity:
        violations.append(
            CompatibilityViolation("task_definition_mismatch", "task_definition")
        )
    if left.evaluation_policy != right.evaluation_policy:
        violations.append(
            CompatibilityViolation("evaluation_policy_mismatch", "evaluation_policy")
        )
    if left.objectives != right.objectives:
        violations.append(CompatibilityViolation("objectives_mismatch", "objectives"))
    return CompatibilityDecision(
        CompatibilityRelationship.COMPARABLE, tuple(violations)
    )


def require_comparable(left: ComparisonProfile, right: ComparisonProfile) -> None:
    check_comparison_compatibility(left, right).require()


def check_merge_compatibility(
    left: CompatibilityGroup,
    right: CompatibilityGroup,
    merge_method: str,
    *,
    left_integrity_verified: bool = False,
    right_integrity_verified: bool = False,
) -> CompatibilityDecision:
    """Check conservative v1 LoRA merge compatibility."""

    require_identifier("merge_method", merge_method)
    if not isinstance(left_integrity_verified, bool) or not isinstance(
        right_integrity_verified, bool
    ):
        raise RecordValidationError("integrity verification flags must be booleans")
    violations: list[CompatibilityViolation] = []
    if left.adapter_type != "lora" or right.adapter_type != "lora":
        violations.append(CompatibilityViolation("adapter_not_lora", "adapter_type"))
    if left.base_model_revision.identity != right.base_model_revision.identity:
        violations.append(
            CompatibilityViolation("base_model_mismatch", "base_model_revision")
        )
    if left.tokenizer_identity != right.tokenizer_identity:
        violations.append(
            CompatibilityViolation("tokenizer_mismatch", "tokenizer_identity")
        )
    if left.adapter_type != right.adapter_type:
        violations.append(
            CompatibilityViolation("adapter_type_mismatch", "adapter_type")
        )
    if left.target_modules != right.target_modules:
        violations.append(
            CompatibilityViolation("target_modules_mismatch", "target_modules")
        )
    if (
        merge_method not in left.merge_methods
        or merge_method not in right.merge_methods
    ):
        violations.append(
            CompatibilityViolation("merge_method_unsupported", "merge_methods")
        )
    if not left_integrity_verified:
        violations.append(
            CompatibilityViolation("left_integrity_unverified", "integrity_evidence")
        )
    if not right_integrity_verified:
        violations.append(
            CompatibilityViolation("right_integrity_unverified", "integrity_evidence")
        )
    return CompatibilityDecision(CompatibilityRelationship.MERGE, tuple(violations))


def require_merge_compatible(
    left: CompatibilityGroup,
    right: CompatibilityGroup,
    merge_method: str,
    *,
    left_integrity_verified: bool = False,
    right_integrity_verified: bool = False,
) -> None:
    check_merge_compatibility(
        left,
        right,
        merge_method,
        left_integrity_verified=left_integrity_verified,
        right_integrity_verified=right_integrity_verified,
    ).require()


def check_resume_compatibility(
    checkpoint: ResumeCheckpoint, request: ResumeRequest
) -> CompatibilityDecision:
    """Require retained bytes and exact manifest, resolution, state, and target."""

    violations: list[CompatibilityViolation] = []
    if not checkpoint.available:
        violations.append(
            CompatibilityViolation("checkpoint_unavailable", "availability")
        )
    if checkpoint.experiment_manifest_identity != request.experiment_manifest_identity:
        violations.append(
            CompatibilityViolation(
                "experiment_manifest_mismatch", "experiment_manifest"
            )
        )
    if checkpoint.recipe_resolution.identity != request.recipe_resolution.identity:
        violations.append(
            CompatibilityViolation("recipe_resolution_mismatch", "recipe_resolution")
        )
    if checkpoint.training_state_identity != request.training_state_identity:
        violations.append(
            CompatibilityViolation("training_state_mismatch", "training_state")
        )
    if checkpoint.execution_target.identity != request.execution_target.identity:
        violations.append(
            CompatibilityViolation("execution_target_mismatch", "execution_target")
        )
    return CompatibilityDecision(CompatibilityRelationship.RESUME, tuple(violations))


def require_resume_compatible(
    checkpoint: ResumeCheckpoint, request: ResumeRequest
) -> None:
    check_resume_compatibility(checkpoint, request).require()


def check_deployment_compatibility(
    artifact: Artifact,
    group: CompatibilityGroup,
    target: ExecutionTarget,
    *,
    integrity_evidence: ContentIdentity,
) -> CompatibilityDecision:
    """Check one exact artifact against one declared runtime target."""

    if not isinstance(artifact, Artifact):
        raise RecordValidationError("artifact must be an Artifact")
    if not isinstance(integrity_evidence, ContentIdentity):
        raise RecordValidationError("integrity_evidence must be a content identity")
    violations: list[CompatibilityViolation] = []
    group_identities = {
        reference.identity for reference in artifact.compatibility_groups
    }
    if group.identity not in group_identities:
        violations.append(
            CompatibilityViolation(
                "compatibility_group_mismatch", "compatibility_groups"
            )
        )
    if artifact.base_model_revision.identity != group.base_model_revision.identity:
        violations.append(
            CompatibilityViolation("base_model_mismatch", "base_model_revision")
        )
    if artifact.tokenizer_identity != group.tokenizer_identity:
        violations.append(
            CompatibilityViolation("tokenizer_mismatch", "tokenizer_identity")
        )
    if artifact.adapter_type != group.adapter_type:
        violations.append(
            CompatibilityViolation("adapter_type_mismatch", "adapter_type")
        )
    violations.extend(check_runtime_target_compatibility(group, target).violations)
    if artifact.integrity_evidence != integrity_evidence:
        violations.append(
            CompatibilityViolation("integrity_evidence_mismatch", "integrity_evidence")
        )
    return CompatibilityDecision(
        CompatibilityRelationship.DEPLOYMENT, tuple(violations)
    )


def check_runtime_target_compatibility(
    group: CompatibilityGroup,
    target: ExecutionTarget,
) -> CompatibilityDecision:
    """Check only structural group/target compatibility, not an artifact."""

    violations: list[CompatibilityViolation] = []
    constraint = group.target_constraint(target.target_class)
    if constraint is None:
        violations.append(
            CompatibilityViolation("runtime_target_undeclared", "target_class")
        )
    else:
        if constraint.accelerator_backend != target.accelerator_backend:
            violations.append(
                CompatibilityViolation(
                    "accelerator_backend_mismatch", "accelerator_backend"
                )
            )
        if constraint.runtime_contract != target.runtime_contract:
            violations.append(
                CompatibilityViolation("runtime_contract_mismatch", "runtime_contract")
            )
        if set(constraint.required_capabilities) - set(target.capabilities):
            violations.append(
                CompatibilityViolation("runtime_capability_missing", "capabilities")
            )
    return CompatibilityDecision(
        CompatibilityRelationship.DEPLOYMENT, tuple(violations)
    )


def require_deployment_compatible(
    artifact: Artifact,
    group: CompatibilityGroup,
    target: ExecutionTarget,
    *,
    integrity_evidence: ContentIdentity,
) -> None:
    check_deployment_compatibility(
        artifact,
        group,
        target,
        integrity_evidence=integrity_evidence,
    ).require()
