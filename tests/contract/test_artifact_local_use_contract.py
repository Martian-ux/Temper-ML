import hashlib

import pytest

from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactAvailability,
    ArtifactContentKind,
    AvailabilityState,
    StorageReference,
)
from temper_ml.domain.local_use import AdapterExport, LocalUseSession
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    record_reference,
)


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(kind: str, logical_id: str, revision: str = "v1") -> RecordReference:
    return RecordReference(
        kind, logical_id, _identity(f"{kind}:{logical_id}:{revision}")
    )


def _artifact() -> Artifact:
    return Artifact(
        artifact_id="artifact-adapter-v1",
        project=_reference("project", "project-rewrite"),
        producing_run=_reference("run", "run-attempt-1"),
        adapter_type="lora",
        content_kind=ArtifactContentKind.BUNDLE,
        content_identity=_identity("adapter-bundle"),
        base_model_revision=_reference("base_model_revision", "model-alpha"),
        tokenizer_identity=_identity("tokenizer"),
        compatibility_groups=(_reference("compatibility_group", "group-alpha"),),
        parent_artifacts=(),
        storage_references=(StorageReference("project_store", "adapter_primary"),),
        integrity_evidence=_identity("integrity-evidence"),
        provenance=_identity("provenance"),
        lineage_evidence=_identity("lineage"),
    )


def test_artifact_availability_is_separate_and_fails_closed_after_removal() -> None:
    artifact = _artifact()
    artifact_ref = record_reference(artifact, artifact.artifact_id)
    available = ArtifactAvailability(
        availability_id="availability-v1",
        artifact=artifact_ref,
        state=AvailabilityState.AVAILABLE,
        available_byte_classes=("final_adapter", "checkpoint"),
        storage_references=(StorageReference("project_store", "adapter_primary"),),
        checkpoint_resumable=True,
        observed_content_identity=artifact.content_identity,
    )
    removed = ArtifactAvailability(
        availability_id="availability-v2",
        artifact=artifact_ref,
        state=AvailabilityState.REMOVED,
        available_byte_classes=(),
        storage_references=(),
        checkpoint_resumable=False,
        observed_content_identity=artifact.content_identity,
        supersedes=record_reference(available, available.availability_id),
    )

    assert removed.identity != available.identity
    assert artifact.storage_references
    with pytest.raises(RecordValidationError, match="cannot advertise"):
        ArtifactAvailability(
            availability_id="availability-invalid",
            artifact=artifact_ref,
            state=AvailabilityState.REMOVED,
            available_byte_classes=("checkpoint",),
            storage_references=(),
            checkpoint_resumable=False,
            observed_content_identity=artifact.content_identity,
        )


def test_saved_local_use_session_pins_inputs_outputs_runtime_and_integrity() -> None:
    artifact = _artifact()
    prompt = {"text": "Synthetic prompt"}
    output = {"text": "Synthetic response"}
    session = LocalUseSession(
        session_id="session-local-v1",
        project=artifact.project,
        artifact=record_reference(artifact, artifact.artifact_id),
        artifact_content_identity=artifact.content_identity,
        base_model_revision=artifact.base_model_revision,
        tokenizer_identity=artifact.tokenizer_identity,
        compatibility_group=artifact.compatibility_groups[0],
        execution_target=_reference("execution_target", "target-wsl2-rocm"),
        inference_settings={"temperature": 0, "maximum_tokens": 64},
        inputs=(prompt,),
        outputs=(output,),
        runtime_evidence=_identity("runtime-evidence"),
        integrity_evidence=artifact.integrity_evidence,
    )
    before = session.identity
    prompt["text"] = "Changed after construction"
    output["text"] = "Changed after construction"

    assert session.identity == before
    assert session.to_payload()["inputs"] == [{"text": "Synthetic prompt"}]
    assert session.to_payload()["evaluation_captures"] == []

    with pytest.raises(RecordValidationError, match="equal length"):
        LocalUseSession(
            **{
                **session.__dict__,
                "outputs": (
                    {"text": "one"},
                    {"text": "two"},
                ),
            }
        )


def test_adapter_export_binds_integrity_and_never_implies_deployment() -> None:
    artifact = _artifact()
    exported = AdapterExport(
        export_id="export-adapter-v1",
        artifact=record_reference(artifact, artifact.artifact_id),
        adapter_content_identity=artifact.content_identity,
        integrity_manifest_identity=_identity("export-manifest"),
        integrity_evidence=artifact.integrity_evidence,
        compatibility_groups=artifact.compatibility_groups,
        compatibility_requirements={
            "adapter_type": "lora",
            "target_class": "wsl2_rocm",
        },
        provenance=artifact.provenance,
        export_format="temper_adapter_bundle",
        storage_reference=StorageReference("export_store", "adapter_export_v1"),
    )
    payload = exported.to_payload()

    assert payload["integrity_evidence"] == {
        "algorithm": "sha256",
        "value": artifact.integrity_evidence.value,
    }
    assert "deployment" not in payload
    assert "readiness" not in payload
