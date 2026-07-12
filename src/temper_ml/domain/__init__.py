"""Public Temper-owned domain contracts and primitives."""

from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactAvailability,
    ArtifactContentKind,
    AvailabilityState,
    StorageReference,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import (
    ComparisonProfile,
    CompatibilityDecision,
    CompatibilityError,
    CompatibilityGroup,
    ResumeCheckpoint,
    ResumeRequest,
    RuntimeTargetConstraint,
    check_comparison_compatibility,
    check_deployment_compatibility,
    check_merge_compatibility,
    check_resume_compatibility,
    check_runtime_target_compatibility,
)
from temper_ml.domain.experiments import (
    Experiment,
    ExperimentDerivation,
    ManifestChange,
    ManifestDiff,
    ReproductionMode,
    derive_experiment,
)
from temper_ml.domain.hardware import (
    ExecutionTarget,
    HardwareCapabilityProfile,
    HardwareRequirements,
)
from temper_ml.domain.local_use import AdapterExport, LocalUseSession
from temper_ml.domain.policies import (
    BaselinePolicy,
    FixedReferenceBaseline,
    PerModelBaseline,
    ProjectChampionBaseline,
)
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.recipes import Recipe, RecipeResolution
from temper_ml.domain.records import (
    CORE_LOGICAL_ID_FIELDS,
    CORE_PROJECTION_REGISTRY,
    RecordEnvelope,
    RecordReference,
    RecordValidationError,
    record_reference,
)
from temper_ml.domain.runs import Run
from temper_ml.domain.tasks import TaskDefinition

__all__ = [
    "AdapterExport",
    "Artifact",
    "ArtifactAvailability",
    "ArtifactContentKind",
    "AvailabilityState",
    "BaseModelRevision",
    "BaselinePolicy",
    "CORE_LOGICAL_ID_FIELDS",
    "CORE_PROJECTION_REGISTRY",
    "ComparisonProfile",
    "CompatibilityDecision",
    "CompatibilityError",
    "CompatibilityGroup",
    "ExecutionTarget",
    "Experiment",
    "ExperimentDerivation",
    "FixedReferenceBaseline",
    "HardwareCapabilityProfile",
    "HardwareRequirements",
    "LocalUseSession",
    "ManifestChange",
    "ManifestDiff",
    "PerModelBaseline",
    "Project",
    "ProjectChampionBaseline",
    "ProjectPolicy",
    "Recipe",
    "RecipeResolution",
    "RecordEnvelope",
    "RecordReference",
    "RecordValidationError",
    "ReproductionMode",
    "ResumeCheckpoint",
    "ResumeRequest",
    "Run",
    "RuntimeTargetConstraint",
    "StorageReference",
    "TaskDefinition",
    "check_comparison_compatibility",
    "check_deployment_compatibility",
    "check_merge_compatibility",
    "check_resume_compatibility",
    "check_runtime_target_compatibility",
    "derive_experiment",
    "record_reference",
]
