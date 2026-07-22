"""Fail-closed integrity verification for normalized adapter artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from temper_ml.domain.artifacts import (
    BundleManifest,
    build_bundle_manifest,
    build_bytes_bundle_manifest,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import CompatibilityGroup
from temper_ml.domain.datasets import DatasetVersion
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordValidationError,
    identity_fields,
    parse_identity,
    record_reference,
)
from temper_ml.domain.runs import ResolvedRuntimeRequest, Run
from temper_ml.runtime.fixture_adapter import (
    FIXTURE_ARTIFACT_MEMBERS,
    FIXTURE_RUNTIME_IDENTITY,
    fixture_adapter_bytes,
    fixture_training_state_identity,
)
from temper_ml.runtime.library_adapter import library_training_state_identity
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes

ARTIFACT_INTEGRITY_PROJECTION = HashProjection("artifact.integrity_evidence", "v1")
EXPORT_INTEGRITY_PROJECTION = HashProjection("artifact.export_integrity", "v1")
EXPORT_MANIFEST_MEMBER = "integrity-manifest.json"
EXPORT_BUNDLE_PREFIX = "bundle"


class ArtifactIntegrityError(RuntimeError):
    """A bounded artifact failure that never exposes content or local paths."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ArtifactIntegrityExpectation:
    """Exact Temper-owned facts expected at artifact ingestion."""

    bundle_identity: ContentIdentity
    producing_run: Run
    runtime_request: ResolvedRuntimeRequest
    experiment: Experiment
    recipe_resolution: RecipeResolution
    dataset_version: DatasetVersion
    base_model_revision: BaseModelRevision
    compatibility_group: CompatibilityGroup

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_identity, ContentIdentity):
            raise ArtifactIntegrityError("artifact_expectation_invalid")
        if not isinstance(self.producing_run, Run) or not isinstance(
            self.runtime_request, ResolvedRuntimeRequest
        ):
            raise ArtifactIntegrityError("artifact_expectation_invalid")
        if not isinstance(self.experiment, Experiment) or not isinstance(
            self.recipe_resolution, RecipeResolution
        ):
            raise ArtifactIntegrityError("artifact_expectation_invalid")
        if not isinstance(self.dataset_version, DatasetVersion):
            raise ArtifactIntegrityError("artifact_expectation_invalid")
        if not isinstance(
            self.base_model_revision, BaseModelRevision
        ) or not isinstance(self.compatibility_group, CompatibilityGroup):
            raise ArtifactIntegrityError("artifact_expectation_invalid")
        if self.producing_run.experiment != record_reference(self.experiment):
            raise ArtifactIntegrityError("artifact_run_experiment_mismatch")
        if self.producing_run.request_identity != self.runtime_request.identity:
            raise ArtifactIntegrityError("artifact_run_request_mismatch")
        if (
            self.producing_run.runtime_identity != self.runtime_request.runtime_identity
            or self.producing_run.execution_target
            != self.runtime_request.execution_target
            or self.producing_run.training_state_identity
            != self.runtime_request.training_state_identity
        ):
            raise ArtifactIntegrityError("artifact_run_request_mismatch")
        if (
            self.runtime_request.experiment != record_reference(self.experiment)
            or self.runtime_request.experiment_manifest_identity
            != self.experiment.manifest_identity
        ):
            raise ArtifactIntegrityError("artifact_request_experiment_mismatch")
        if self.runtime_request.recipe_resolution != record_reference(
            self.recipe_resolution
        ):
            raise ArtifactIntegrityError("artifact_request_resolution_mismatch")
        if self.experiment.dataset_version != self.dataset_version.identity:
            raise ArtifactIntegrityError("artifact_dataset_mismatch")
        if (
            self.runtime_request.dataset_version_identity
            != self.dataset_version.identity
        ):
            raise ArtifactIntegrityError("artifact_dataset_mismatch")
        if self.experiment.base_model_revision != record_reference(
            self.base_model_revision
        ):
            raise ArtifactIntegrityError("artifact_base_model_mismatch")
        if (
            self.experiment.tokenizer_identity
            != self.base_model_revision.tokenizer_identity
        ):
            raise ArtifactIntegrityError("artifact_tokenizer_mismatch")
        if self.experiment.compatibility_group != record_reference(
            self.compatibility_group
        ):
            raise ArtifactIntegrityError("artifact_compatibility_group_mismatch")
        if self.compatibility_group.base_model_revision != record_reference(
            self.base_model_revision
        ):
            raise ArtifactIntegrityError("artifact_compatibility_group_mismatch")
        if (
            self.compatibility_group.tokenizer_identity
            != self.base_model_revision.tokenizer_identity
        ):
            raise ArtifactIntegrityError("artifact_compatibility_group_mismatch")


@dataclass(frozen=True)
class ArtifactIntegrityResult:
    bundle_manifest: BundleManifest
    adapter_identity: ContentIdentity
    adapter_config_identity: ContentIdentity
    provenance_identity: ContentIdentity
    evidence_identity: ContentIdentity

    def to_receipt(self) -> dict[str, object]:
        return {
            "bundle_identity": identity_fields(self.bundle_manifest.identity),
            "adapter_identity": identity_fields(self.adapter_identity),
            "adapter_config_identity": identity_fields(self.adapter_config_identity),
            "provenance_identity": identity_fields(self.provenance_identity),
            "integrity_evidence": identity_fields(self.evidence_identity),
            "structure_verified": True,
            "provenance_verified": True,
            "compatibility_verified": True,
        }


@dataclass(frozen=True)
class ExportIntegrityResult:
    export_bundle_manifest: BundleManifest
    integrity_manifest_identity: ContentIdentity
    artifact_integrity: ArtifactIntegrityResult
    evidence_identity: ContentIdentity

    def to_receipt(self) -> dict[str, object]:
        return {
            "export_bundle_identity": identity_fields(
                self.export_bundle_manifest.identity
            ),
            "integrity_manifest_identity": identity_fields(
                self.integrity_manifest_identity
            ),
            "artifact_bundle_identity": identity_fields(
                self.artifact_integrity.bundle_manifest.identity
            ),
            "integrity_evidence": identity_fields(self.evidence_identity),
            "deployment_ready": False,
            "hosted_deployment": False,
        }


def verify_artifact_bundle(
    root: Path | str,
    expectation: ArtifactIntegrityExpectation,
) -> ArtifactIntegrityResult:
    """Verify a complete transferred bundle against Temper-owned facts."""

    if not isinstance(expectation, ArtifactIntegrityExpectation):
        raise ArtifactIntegrityError("artifact_expectation_invalid")
    bundle_root = Path(root)
    try:
        manifest = build_bundle_manifest(bundle_root)
    except (OSError, ValueError):
        raise ArtifactIntegrityError("artifact_bundle_unreadable") from None
    if tuple(member.path for member in manifest.members) != FIXTURE_ARTIFACT_MEMBERS:
        raise ArtifactIntegrityError("artifact_structure_mismatch")
    if manifest.identity != expectation.bundle_identity:
        raise ArtifactIntegrityError("artifact_content_identity_mismatch")

    members = _read_members(bundle_root, FIXTURE_ARTIFACT_MEMBERS)
    if build_bytes_bundle_manifest(members) != manifest:
        raise ArtifactIntegrityError("artifact_changed_during_verification")
    adapter = members["adapter.bin"]
    config = _canonical_object(
        members["adapter_config.json"], "artifact_config_invalid"
    )
    provenance = _canonical_object(
        members["provenance.json"], "artifact_provenance_invalid"
    )
    adapter_identity = _bytes_identity(adapter)
    if expectation.runtime_request.runtime_identity == FIXTURE_RUNTIME_IDENTITY:
        expected_adapter = fixture_adapter_bytes(
            expectation.experiment,
            expectation.recipe_resolution,
            expectation.dataset_version,
        )
        if adapter != expected_adapter:
            raise ArtifactIntegrityError("artifact_adapter_bytes_mismatch")
    _verify_config(config, adapter_identity, expectation)
    _verify_provenance(provenance, expectation)
    config_identity = _bytes_identity(members["adapter_config.json"])
    provenance_identity = _bytes_identity(members["provenance.json"])
    evidence = content_identity(
        ARTIFACT_INTEGRITY_PROJECTION,
        {
            "schema_version": "v1",
            "bundle_identity": identity_fields(manifest.identity),
            "adapter_identity": identity_fields(adapter_identity),
            "adapter_config_identity": identity_fields(config_identity),
            "provenance_identity": identity_fields(provenance_identity),
            "producing_run": record_reference(expectation.producing_run).to_dict(),
            "resolved_runtime_request": record_reference(
                expectation.runtime_request
            ).to_dict(),
            "base_model_revision": record_reference(
                expectation.base_model_revision
            ).to_dict(),
            "tokenizer_identity": identity_fields(
                expectation.base_model_revision.tokenizer_identity
            ),
            "compatibility_group": record_reference(
                expectation.compatibility_group
            ).to_dict(),
            "structure_verified": True,
            "provenance_verified": True,
            "compatibility_verified": True,
        },
    )
    return ArtifactIntegrityResult(
        manifest,
        adapter_identity,
        config_identity,
        provenance_identity,
        evidence,
    )


def export_integrity_manifest_bytes(
    artifact_reference: Mapping[str, object],
    artifact_integrity: ArtifactIntegrityResult,
    compatibility_requirements: Mapping[str, object],
) -> bytes:
    """Create the portable integrity manifest included beside exported bytes."""

    if not isinstance(artifact_integrity, ArtifactIntegrityResult):
        raise ArtifactIntegrityError("export_integrity_input_invalid")
    if not isinstance(artifact_reference, Mapping) or not isinstance(
        compatibility_requirements, Mapping
    ):
        raise ArtifactIntegrityError("export_integrity_input_invalid")
    return dumps_canonical_json(
        {
            "schema_version": "v1",
            "artifact": dict(artifact_reference),
            "artifact_bundle_identity": identity_fields(
                artifact_integrity.bundle_manifest.identity
            ),
            "adapter_identity": identity_fields(artifact_integrity.adapter_identity),
            "artifact_integrity_evidence": identity_fields(
                artifact_integrity.evidence_identity
            ),
            "compatibility_requirements": dict(compatibility_requirements),
            "members": [
                member.projected_fields()
                for member in artifact_integrity.bundle_manifest.members
            ],
            "provenance_identity": identity_fields(
                artifact_integrity.provenance_identity
            ),
            "hosted_deployment": False,
            "deployment_ready": False,
        }
    )


def verify_export_bundle(
    root: Path | str,
    expectation: ArtifactIntegrityExpectation,
    *,
    expected_artifact_reference: Mapping[str, object],
    expected_compatibility_requirements: Mapping[str, object],
    expected_integrity_manifest_identity: ContentIdentity | None = None,
) -> ExportIntegrityResult:
    """Re-verify exported adapter bytes and their separate integrity manifest."""

    export_root = Path(root)
    expected_paths = tuple(
        sorted(
            (
                EXPORT_MANIFEST_MEMBER,
                *(
                    f"{EXPORT_BUNDLE_PREFIX}/{path}"
                    for path in FIXTURE_ARTIFACT_MEMBERS
                ),
            )
        )
    )
    try:
        export_manifest = build_bundle_manifest(export_root)
    except (OSError, ValueError):
        raise ArtifactIntegrityError("export_bundle_unreadable") from None
    if tuple(member.path for member in export_manifest.members) != expected_paths:
        raise ArtifactIntegrityError("export_structure_mismatch")
    artifact_integrity = verify_artifact_bundle(
        export_root / EXPORT_BUNDLE_PREFIX, expectation
    )
    try:
        integrity_bytes = read_stable_bytes(export_root / EXPORT_MANIFEST_MEMBER)
    except SafeIoError:
        raise ArtifactIntegrityError("export_manifest_unreadable") from None
    integrity_identity = _bytes_identity(integrity_bytes)
    if (
        expected_integrity_manifest_identity is not None
        and integrity_identity != expected_integrity_manifest_identity
    ):
        raise ArtifactIntegrityError("export_manifest_identity_mismatch")
    expected_bytes = export_integrity_manifest_bytes(
        expected_artifact_reference,
        artifact_integrity,
        expected_compatibility_requirements,
    )
    if integrity_bytes != expected_bytes:
        raise ArtifactIntegrityError("export_manifest_content_mismatch")
    final_manifest = build_bundle_manifest(export_root)
    if final_manifest != export_manifest:
        raise ArtifactIntegrityError("export_changed_during_verification")
    evidence = content_identity(
        EXPORT_INTEGRITY_PROJECTION,
        {
            "schema_version": "v1",
            "export_bundle_identity": identity_fields(export_manifest.identity),
            "integrity_manifest_identity": identity_fields(integrity_identity),
            "artifact_integrity_evidence": identity_fields(
                artifact_integrity.evidence_identity
            ),
            "hosted_deployment": False,
            "deployment_ready": False,
        },
    )
    return ExportIntegrityResult(
        export_manifest,
        integrity_identity,
        artifact_integrity,
        evidence,
    )


def _read_members(root: Path, paths: tuple[str, ...]) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for path in paths:
        try:
            members[path] = read_stable_bytes(root / path)
        except SafeIoError:
            raise ArtifactIntegrityError("artifact_member_unreadable") from None
    return members


def _canonical_object(data: bytes, code: str) -> dict[str, Any]:
    try:
        value = loads_canonical_json(data)
        if not isinstance(value, dict) or dumps_canonical_json(value) != data:
            raise ArtifactIntegrityError(code)
        return value
    except ArtifactIntegrityError:
        raise
    except (CanonicalJsonError, UnicodeError, TypeError, ValueError):
        raise ArtifactIntegrityError(code) from None


def _verify_config(
    value: Mapping[str, Any],
    adapter_identity: ContentIdentity,
    expectation: ArtifactIntegrityExpectation,
) -> None:
    common = {
        "schema_version",
        "adapter_type",
        "target_modules",
        "rank",
        "alpha",
        "base_model_revision",
        "tokenizer_identity",
        "compatibility_group",
        "adapter_identity",
        "runtime_identity",
        "training_steps",
    }
    fixture = expectation.runtime_request.runtime_identity == FIXTURE_RUNTIME_IDENTITY
    library = {
        "runtime_kind",
        "adapter_payload_format",
        "library_versions",
        "library_adapter_config",
    }
    required = common if fixture else common | library
    if set(value) != required or value["schema_version"] != "v1":
        raise ArtifactIntegrityError("artifact_config_invalid")
    resolution = expectation.recipe_resolution
    expected = {
        "adapter_type": resolution.adapter_type,
        "target_modules": list(resolution.target_modules),
        "rank": resolution.rank,
        "alpha": resolution.alpha,
        "base_model_revision": record_reference(
            expectation.base_model_revision
        ).to_dict(),
        "tokenizer_identity": identity_fields(
            expectation.base_model_revision.tokenizer_identity
        ),
        "compatibility_group": record_reference(
            expectation.compatibility_group
        ).to_dict(),
        "adapter_identity": identity_fields(adapter_identity),
        "runtime_identity": identity_fields(
            expectation.runtime_request.runtime_identity
        ),
        "training_steps": resolution.training_steps,
    }
    if any(value.get(field) != expected[field] for field in expected):
        raise ArtifactIntegrityError("artifact_config_mismatch")
    if (
        expectation.compatibility_group.adapter_type != resolution.adapter_type
        or expectation.compatibility_group.target_modules != resolution.target_modules
    ):
        raise ArtifactIntegrityError("artifact_compatibility_group_mismatch")
    if fixture:
        return
    if value.get("runtime_kind") != "library":
        raise ArtifactIntegrityError("artifact_config_mismatch")
    payload_format = value.get("adapter_payload_format")
    if payload_format != "safetensors":
        raise ArtifactIntegrityError("artifact_config_invalid")
    versions = value.get("library_versions")
    if not isinstance(versions, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        or "://" in version
        or version.startswith(("/", "\\\\"))
        for name, version in versions.items()
    ):
        raise ArtifactIntegrityError("artifact_config_invalid")
    declared_versions = expectation.recipe_resolution.library_versions
    if any(
        name not in versions or versions[name] != declared
        for name, declared in declared_versions.items()
    ):
        raise ArtifactIntegrityError("artifact_config_mismatch")
    library_config = value.get("library_adapter_config")
    expected_library_config = {
        "peft_type": "lora",
        "task_type": "causal_lm",
        "bias": "none",
        "rank": resolution.rank,
        "alpha": resolution.alpha,
        "dropout": str(resolution.dropout),
        "payload_format": payload_format,
    }
    if library_config != expected_library_config:
        raise ArtifactIntegrityError("artifact_config_mismatch")


def _verify_provenance(
    value: Mapping[str, Any], expectation: ArtifactIntegrityExpectation
) -> None:
    required = {
        "schema_version",
        "producing_run",
        "resolved_runtime_request",
        "experiment",
        "experiment_manifest_identity",
        "recipe_resolution",
        "dataset_version_identity",
        "final_training_state_identity",
    }
    if set(value) != required or value["schema_version"] != "v1":
        raise ArtifactIntegrityError("artifact_provenance_invalid")
    expected = {
        "producing_run": record_reference(expectation.producing_run).to_dict(),
        "resolved_runtime_request": record_reference(
            expectation.runtime_request
        ).to_dict(),
        "experiment": record_reference(expectation.experiment).to_dict(),
        "experiment_manifest_identity": identity_fields(
            expectation.experiment.manifest_identity
        ),
        "recipe_resolution": record_reference(expectation.recipe_resolution).to_dict(),
        "dataset_version_identity": identity_fields(
            expectation.runtime_request.dataset_version_identity
        ),
    }
    if any(value.get(field) != expected[field] for field in expected):
        raise ArtifactIntegrityError("artifact_provenance_mismatch")
    raw_state = value.get("final_training_state_identity")
    if not isinstance(raw_state, Mapping):
        raise ArtifactIntegrityError("artifact_provenance_invalid")
    try:
        observed_state = parse_identity(
            raw_state, field="final_training_state_identity"
        )
    except RecordValidationError:
        raise ArtifactIntegrityError("artifact_provenance_invalid") from None
    if expectation.runtime_request.runtime_identity == FIXTURE_RUNTIME_IDENTITY:
        expected_state = fixture_training_state_identity(
            expectation.experiment,
            expectation.recipe_resolution,
            expectation.dataset_version,
            expectation.recipe_resolution.training_steps,
        )
    else:
        expected_state = library_training_state_identity(
            expectation.experiment,
            expectation.recipe_resolution,
            expectation.dataset_version,
            expectation.runtime_request.runtime_identity,
            expectation.recipe_resolution.training_steps,
        )
    if observed_state != expected_state:
        raise ArtifactIntegrityError("artifact_provenance_mismatch")


def _bytes_identity(data: bytes) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(data).hexdigest())
