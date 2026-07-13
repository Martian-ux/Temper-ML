"""Verified focused local use, local batch inference, and portable export."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from temper_ml.app_services._records import (
    require_no_conflicting_logical_revision,
    write_record_idempotently,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactAvailability,
    AvailabilityState,
    StorageReference,
)
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.compatibility import (
    CompatibilityGroup,
    check_runtime_target_compatibility,
)
from temper_ml.domain.datasets import DatasetVersion
from temper_ml.domain.experiments import Experiment
from temper_ml.domain.hardware import ExecutionTarget
from temper_ml.domain.local_use import AdapterExport, LocalUseSession
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    TypedRecord,
    identity_fields,
    record_reference,
    require_identifier,
    thaw_json,
)
from temper_ml.domain.runs import ResolvedRuntimeRequest, Run
from temper_ml.filesystem import UnsafeFilesystemPath, ensure_safe_directory
from temper_ml.runtime.artifact_integrity import (
    EXPORT_BUNDLE_PREFIX,
    EXPORT_MANIFEST_MEMBER,
    ArtifactIntegrityError,
    ArtifactIntegrityExpectation,
    ArtifactIntegrityResult,
    ExportIntegrityResult,
    export_integrity_manifest_bytes,
    verify_artifact_bundle,
    verify_export_bundle,
)
from temper_ml.runtime.fixture_adapter import FIXTURE_ARTIFACT_MEMBERS
from temper_ml.runtime.fixture_inference import (
    FixtureInferenceError,
    FixtureInferenceRequest,
    FixtureInferenceResult,
    FixtureInferenceRuntime,
    InferenceSettings,
)
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes, write_once_bytes

RUNTIME_OUTPUT_DIRECTORY = ".temper-fixture-output"


@dataclass(frozen=True)
class LocalUseRequest:
    """Exact verified artifact, runtime target, inputs, and persistence choice."""

    artifact: Artifact
    base_model_revision: BaseModelRevision
    compatibility_group: CompatibilityGroup
    execution_target: ExecutionTarget
    settings: InferenceSettings
    inputs: tuple[Mapping[str, Any], ...]
    session_id: str | None = None

    def __post_init__(self) -> None:
        expected = (
            (self.artifact, Artifact),
            (self.base_model_revision, BaseModelRevision),
            (self.compatibility_group, CompatibilityGroup),
            (self.execution_target, ExecutionTarget),
            (self.settings, InferenceSettings),
        )
        if any(not isinstance(value, kind) for value, kind in expected):
            raise ApplicationServiceError("local_use_request_invalid")
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ApplicationServiceError("local_use_request_invalid")
        if self.session_id is not None:
            try:
                require_identifier("session_id", self.session_id)
            except RecordValidationError:
                raise ApplicationServiceError("local_use_request_invalid") from None

    @property
    def ephemeral(self) -> bool:
        return self.session_id is None


@dataclass(frozen=True)
class LocalUseResult:
    inference: FixtureInferenceResult
    integrity: ArtifactIntegrityResult
    session: LocalUseSession | None

    @property
    def ephemeral(self) -> bool:
        return self.session is None

    def to_view(self) -> dict[str, object]:
        value = self.inference.to_view()
        value["ephemeral"] = self.ephemeral
        value["saved_canonical_session"] = self.session is not None
        value["artifact_integrity"] = self.integrity.to_receipt()
        if self.session is not None:
            value["session"] = record_reference(self.session).to_dict()
        return value


@dataclass(frozen=True)
class AdapterExportRequest:
    export_id: str
    artifact: Artifact
    base_model_revision: BaseModelRevision
    compatibility_group: CompatibilityGroup
    execution_target: ExecutionTarget

    def __post_init__(self) -> None:
        try:
            require_identifier("export_id", self.export_id)
        except RecordValidationError:
            raise ApplicationServiceError("adapter_export_request_invalid") from None
        expected = (
            (self.artifact, Artifact),
            (self.base_model_revision, BaseModelRevision),
            (self.compatibility_group, CompatibilityGroup),
            (self.execution_target, ExecutionTarget),
        )
        if any(not isinstance(value, kind) for value, kind in expected):
            raise ApplicationServiceError("adapter_export_request_invalid")


@dataclass(frozen=True)
class VerifiedAdapterExport:
    record: AdapterExport
    integrity: ExportIntegrityResult
    local_root: Path

    def to_view(self) -> dict[str, object]:
        return {
            "status": "verified",
            "adapter_export": record_reference(self.record).to_dict(),
            "integrity": self.integrity.to_receipt(),
            "hosted_deployment": False,
            "deployment_ready": False,
        }


@dataclass(frozen=True)
class _ArtifactContext:
    artifact: Artifact
    availability: ArtifactAvailability
    producing_run: Run
    runtime_request: ResolvedRuntimeRequest
    experiment: Experiment
    recipe_resolution: RecipeResolution
    dataset_version: DatasetVersion
    base_model_revision: BaseModelRevision
    compatibility_group: CompatibilityGroup
    integrity: ArtifactIntegrityResult
    root: Path

    @property
    def expectation(self) -> ArtifactIntegrityExpectation:
        return ArtifactIntegrityExpectation(
            bundle_identity=self.artifact.content_identity,
            producing_run=self.producing_run,
            runtime_request=self.runtime_request,
            experiment=self.experiment,
            recipe_resolution=self.recipe_resolution,
            dataset_version=self.dataset_version,
            base_model_revision=self.base_model_revision,
            compatibility_group=self.compatibility_group,
        )


class LocalUseService:
    """Re-verify an artifact before every local inference or export operation."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        runtime: FixtureInferenceRuntime | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.store = TypedEvidenceStore(self.project_root)
        self.runtime = runtime if runtime is not None else FixtureInferenceRuntime()
        if not isinstance(self.runtime, FixtureInferenceRuntime):
            raise ApplicationServiceError("fixture_inference_runtime_invalid")

    def focused(self, request: LocalUseRequest) -> LocalUseResult:
        """Run one focused local-use input after full compatibility verification."""

        if not isinstance(request, LocalUseRequest) or len(request.inputs) != 1:
            raise ApplicationServiceError("focused_local_use_requires_one_input")
        return self._infer(request)

    def batch(self, request: LocalUseRequest) -> LocalUseResult:
        """Run one deterministic local batch with settings shared by every input."""

        if not isinstance(request, LocalUseRequest):
            raise ApplicationServiceError("local_use_request_invalid")
        return self._infer(request)

    def export(self, request: AdapterExportRequest) -> VerifiedAdapterExport:
        """Write and re-verify a portable bundle without deployment semantics."""

        if not isinstance(request, AdapterExportRequest):
            raise ApplicationServiceError("adapter_export_request_invalid")
        context = self._verified_artifact(
            request.artifact,
            request.base_model_revision,
            request.compatibility_group,
            request.execution_target,
        )
        requirements = self._compatibility_requirements(
            context, request.execution_target
        )
        root = self._export_root(request.export_id)
        bundle_root = root / EXPORT_BUNDLE_PREFIX
        try:
            ensure_safe_directory(bundle_root)
        except (OSError, UnsafeFilesystemPath):
            raise ApplicationServiceError("adapter_export_output_unavailable") from None
        for member in FIXTURE_ARTIFACT_MEMBERS:
            try:
                data = read_stable_bytes(context.root / member)
            except SafeIoError:
                raise ApplicationServiceError(
                    "adapter_export_source_unreadable"
                ) from None
            self._write_idempotent(bundle_root / member, data)
        manifest_bytes = export_integrity_manifest_bytes(
            record_reference(context.artifact).to_dict(),
            context.integrity,
            requirements,
        )
        self._write_idempotent(root / EXPORT_MANIFEST_MEMBER, manifest_bytes)
        manifest_identity = ContentIdentity(
            "sha256", hashlib.sha256(manifest_bytes).hexdigest()
        )
        try:
            integrity = verify_export_bundle(
                root,
                context.expectation,
                expected_artifact_reference=record_reference(
                    context.artifact
                ).to_dict(),
                expected_compatibility_requirements=requirements,
                expected_integrity_manifest_identity=manifest_identity,
            )
        except ArtifactIntegrityError as exc:
            raise ApplicationServiceError(exc.code) from None
        record = AdapterExport(
            export_id=request.export_id,
            artifact=record_reference(context.artifact),
            adapter_content_identity=context.artifact.content_identity,
            integrity_manifest_identity=integrity.integrity_manifest_identity,
            integrity_evidence=integrity.evidence_identity,
            compatibility_groups=context.artifact.compatibility_groups,
            compatibility_requirements=requirements,
            provenance=context.artifact.provenance,
            export_format="temper_fixture_adapter_bundle",
            storage_reference=StorageReference("export_store", request.export_id),
        )
        try:
            require_no_conflicting_logical_revision(
                self.store,
                record,
                conflict_code="adapter_export_conflict",
            )
            write_record_idempotently(
                self.store,
                record,
                conflict_code="adapter_export_conflict",
            )
            self.store.append_event(
                "local-use",
                EventRequest(
                    f"adapter-export-{record.export_id}",
                    "adapter_export_verified",
                    {
                        "adapter_export_identity": identity_fields(record.identity),
                        "artifact_integrity_evidence": identity_fields(
                            context.integrity.evidence_identity
                        ),
                        "export_integrity_evidence": identity_fields(
                            integrity.evidence_identity
                        ),
                        "hosted_deployment": False,
                        "deployment_ready": False,
                    },
                ),
            )
            self.store.verify()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        return VerifiedAdapterExport(record, integrity, root)

    def verify_export(
        self,
        record: AdapterExport,
        *,
        artifact: Artifact,
        base_model_revision: BaseModelRevision,
        compatibility_group: CompatibilityGroup,
        execution_target: ExecutionTarget,
    ) -> ExportIntegrityResult:
        """Re-open a recorded export and fail closed on any changed byte."""

        stored = self._require_exact(record)
        if not isinstance(stored, AdapterExport):
            raise ApplicationServiceError("adapter_export_record_invalid")
        if stored.artifact != record_reference(artifact):
            raise ApplicationServiceError("adapter_export_artifact_mismatch")
        if (
            stored.storage_reference.provider != "export_store"
            or stored.storage_reference.logical_key != stored.export_id
        ):
            raise ApplicationServiceError("adapter_export_storage_invalid")
        context = self._verified_artifact(
            artifact,
            base_model_revision,
            compatibility_group,
            execution_target,
        )
        requirements = self._compatibility_requirements(context, execution_target)
        if thaw_json(stored.compatibility_requirements) != requirements:
            raise ApplicationServiceError("adapter_export_compatibility_mismatch")
        try:
            result = verify_export_bundle(
                self._export_root(stored.export_id),
                context.expectation,
                expected_artifact_reference=record_reference(artifact).to_dict(),
                expected_compatibility_requirements=requirements,
                expected_integrity_manifest_identity=(
                    stored.integrity_manifest_identity
                ),
            )
        except ArtifactIntegrityError as exc:
            raise ApplicationServiceError(exc.code) from None
        if (
            result.evidence_identity != stored.integrity_evidence
            or result.artifact_integrity.evidence_identity
            != context.integrity.evidence_identity
        ):
            raise ApplicationServiceError("adapter_export_integrity_mismatch")
        return result

    def _infer(self, request: LocalUseRequest) -> LocalUseResult:
        context = self._verified_artifact(
            request.artifact,
            request.base_model_revision,
            request.compatibility_group,
            request.execution_target,
        )
        try:
            adapter_bytes = read_stable_bytes(context.root / "adapter.bin")
            if (
                ContentIdentity("sha256", hashlib.sha256(adapter_bytes).hexdigest())
                != context.integrity.adapter_identity
            ):
                raise ApplicationServiceError("local_use_adapter_identity_mismatch")
            inference = self.runtime.infer(
                FixtureInferenceRequest(
                    adapter_bytes=adapter_bytes,
                    artifact_content_identity=context.artifact.content_identity,
                    settings=request.settings,
                    inputs=request.inputs,
                )
            )
        except SafeIoError:
            raise ApplicationServiceError("local_use_adapter_unreadable") from None
        except FixtureInferenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        if request.ephemeral:
            return LocalUseResult(inference, context.integrity, None)
        if request.session_id is None:
            raise ApplicationServiceError("local_use_session_id_missing")
        session = LocalUseSession(
            session_id=request.session_id,
            project=context.artifact.project,
            artifact=record_reference(context.artifact),
            artifact_content_identity=context.artifact.content_identity,
            base_model_revision=record_reference(context.base_model_revision),
            tokenizer_identity=context.base_model_revision.tokenizer_identity,
            compatibility_group=record_reference(context.compatibility_group),
            execution_target=record_reference(request.execution_target),
            inference_settings=request.settings.to_dict(),
            inputs=tuple(thaw_json(value) for value in inference.inputs),
            outputs=tuple(thaw_json(value) for value in inference.outputs),
            runtime_evidence=inference.runtime_evidence,
            integrity_evidence=context.integrity.evidence_identity,
        )
        try:
            require_no_conflicting_logical_revision(
                self.store,
                session,
                conflict_code="local_use_session_conflict",
            )
            write_record_idempotently(
                self.store,
                session,
                conflict_code="local_use_session_conflict",
            )
            self.store.append_event(
                "local-use",
                EventRequest(
                    f"local-use-session-{session.session_id}",
                    "local_use_session_saved",
                    {
                        "session_identity": identity_fields(session.identity),
                        "input_count": len(session.inputs),
                        "ephemeral": False,
                        "integrity_reverified": True,
                    },
                ),
            )
            self.store.verify()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        return LocalUseResult(inference, context.integrity, session)

    def _verified_artifact(
        self,
        artifact: Artifact,
        base_model_revision: BaseModelRevision,
        compatibility_group: CompatibilityGroup,
        execution_target: ExecutionTarget,
    ) -> _ArtifactContext:
        try:
            self.store.verify()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        exact_artifact = self._require_exact(artifact)
        exact_model = self._require_exact(base_model_revision)
        exact_group = self._require_exact(compatibility_group)
        exact_target = self._require_exact(execution_target)
        if not isinstance(exact_artifact, Artifact) or not isinstance(
            exact_model, BaseModelRevision
        ):
            raise ApplicationServiceError("local_use_dependency_invalid")
        if not isinstance(exact_group, CompatibilityGroup) or not isinstance(
            exact_target, ExecutionTarget
        ):
            raise ApplicationServiceError("local_use_dependency_invalid")
        if exact_artifact.base_model_revision != record_reference(exact_model):
            raise ApplicationServiceError("local_use_base_model_mismatch")
        if exact_artifact.tokenizer_identity != exact_model.tokenizer_identity:
            raise ApplicationServiceError("local_use_tokenizer_mismatch")
        if record_reference(exact_group) not in exact_artifact.compatibility_groups:
            raise ApplicationServiceError("local_use_compatibility_group_mismatch")
        if (
            exact_group.base_model_revision != record_reference(exact_model)
            or exact_group.tokenizer_identity != exact_model.tokenizer_identity
        ):
            raise ApplicationServiceError("local_use_compatibility_group_mismatch")
        if not check_runtime_target_compatibility(exact_group, exact_target).compatible:
            raise ApplicationServiceError("local_use_execution_target_incompatible")
        availability = self._current_availability(exact_artifact)
        if availability.state is not AvailabilityState.AVAILABLE:
            raise ApplicationServiceError("local_use_artifact_unavailable")
        if (
            availability.observed_content_identity != exact_artifact.content_identity
            or availability.storage_references != exact_artifact.storage_references
        ):
            raise ApplicationServiceError("local_use_artifact_availability_mismatch")
        if exact_artifact.storage_references != (
            StorageReference("project_store", exact_artifact.artifact_id),
        ):
            raise ApplicationServiceError("local_use_artifact_storage_invalid")
        run = self._resolve(exact_artifact.producing_run, Run)
        runtime_request = self._runtime_request(run.request_identity)
        experiment = self._resolve(run.experiment, Experiment)
        resolution = self._resolve(runtime_request.recipe_resolution, RecipeResolution)
        dataset = self._record_by_identity(
            DatasetVersion, runtime_request.dataset_version_identity
        )
        root = self._artifact_root(exact_artifact.artifact_id)
        expectation = ArtifactIntegrityExpectation(
            bundle_identity=exact_artifact.content_identity,
            producing_run=run,
            runtime_request=runtime_request,
            experiment=experiment,
            recipe_resolution=resolution,
            dataset_version=dataset,
            base_model_revision=exact_model,
            compatibility_group=exact_group,
        )
        try:
            integrity = verify_artifact_bundle(root, expectation)
        except ArtifactIntegrityError as exc:
            raise ApplicationServiceError(exc.code) from None
        if (
            integrity.evidence_identity != exact_artifact.integrity_evidence
            or integrity.provenance_identity != exact_artifact.provenance
            or integrity.bundle_manifest.identity != exact_artifact.content_identity
        ):
            raise ApplicationServiceError("local_use_artifact_integrity_mismatch")
        return _ArtifactContext(
            exact_artifact,
            availability,
            run,
            runtime_request,
            experiment,
            resolution,
            dataset,
            exact_model,
            exact_group,
            integrity,
            root,
        )

    def _current_availability(self, artifact: Artifact) -> ArtifactAvailability:
        values = [
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, ArtifactAvailability)
            and stored.record.artifact == record_reference(artifact)
        ]
        superseded = {
            value.supersedes.identity
            for value in values
            if value.supersedes is not None
        }
        current = [value for value in values if value.identity not in superseded]
        if len(current) != 1:
            raise ApplicationServiceError("artifact_availability_ambiguous")
        return current[0]

    def _require_exact(self, record: TypedRecord) -> TypedRecord:
        try:
            stored = self.store.read_record(record_reference(record))
        except (EvidenceError, RecordValidationError):
            raise ApplicationServiceError("local_use_dependency_missing") from None
        if (
            type(stored.record) is not type(record)
            or stored.envelope.to_dict() != record.to_dict()
        ):
            raise ApplicationServiceError("local_use_dependency_mismatch")
        return stored.record

    def _resolve(self, reference: RecordReference, kind: type[TypedRecord]) -> Any:
        try:
            record = self.store.read_record(reference).record
        except EvidenceError:
            raise ApplicationServiceError("local_use_dependency_missing") from None
        if not isinstance(record, kind):
            raise ApplicationServiceError("local_use_dependency_invalid")
        return record

    def _runtime_request(self, identity: ContentIdentity) -> ResolvedRuntimeRequest:
        return self._record_by_identity(ResolvedRuntimeRequest, identity)

    def _record_by_identity(self, kind: type[Any], identity: ContentIdentity) -> Any:
        matches = [
            stored.record
            for stored in self.store.iter_records()
            if isinstance(stored.record, kind) and stored.record.identity == identity
        ]
        if len(matches) != 1:
            raise ApplicationServiceError("local_use_dependency_missing")
        return matches[0]

    @staticmethod
    def _compatibility_requirements(
        context: _ArtifactContext,
        target: ExecutionTarget,
    ) -> dict[str, object]:
        constraint = context.compatibility_group.target_constraint(target.target_class)
        if constraint is None:
            raise ApplicationServiceError("adapter_export_target_incompatible")
        return {
            "adapter_type": context.artifact.adapter_type,
            "base_model_revision": record_reference(
                context.base_model_revision
            ).to_dict(),
            "tokenizer_identity": identity_fields(
                context.base_model_revision.tokenizer_identity
            ),
            "compatibility_group": record_reference(
                context.compatibility_group
            ).to_dict(),
            "target_modules": list(context.compatibility_group.target_modules),
            "target_class": target.target_class,
            "runtime_contract": identity_fields(target.runtime_contract),
        }

    def _runtime_root(self) -> Path:
        return self.project_root / RUNTIME_OUTPUT_DIRECTORY

    def _artifact_root(self, artifact_id: str) -> Path:
        return self._runtime_root() / "artifacts" / artifact_id

    def _export_root(self, export_id: str) -> Path:
        return self._runtime_root() / "exports" / export_id

    @staticmethod
    def _write_idempotent(path: Path, data: bytes) -> None:
        try:
            write_once_bytes(path, data)
        except FileExistsError:
            try:
                existing = read_stable_bytes(path)
            except SafeIoError:
                raise ApplicationServiceError(
                    "adapter_export_output_unreadable"
                ) from None
            if existing != data:
                raise ApplicationServiceError("adapter_export_output_conflict")
