from dataclasses import replace
import hashlib

import pytest

from temper_ml.domain.artifacts import (
    Artifact,
    ArtifactContentKind,
    StorageReference,
)
from temper_ml.domain.compatibility import (
    ComparisonProfile,
    CompatibilityError,
    CompatibilityGroup,
    ResumeCheckpoint,
    ResumeRequest,
    RuntimeTargetConstraint,
    check_comparison_compatibility,
    check_deployment_compatibility,
    check_merge_compatibility,
    check_resume_compatibility,
)
from temper_ml.domain.hardware import ExecutionTarget
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import RecordReference, record_reference


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(kind: str, logical_id: str, revision: str = "v1") -> RecordReference:
    return RecordReference(
        kind, logical_id, _identity(f"{kind}:{logical_id}:{revision}")
    )


def _target(target_class: str = "wsl2_rocm") -> ExecutionTarget:
    platform = "linux" if target_class == "wsl2_rocm" else "windows"
    return ExecutionTarget(
        target_id=f"target-{target_class}",
        target_class=target_class,
        platform=platform,
        accelerator_backend="rocm",
        runtime_contract=_identity(f"runtime:{target_class}"),
        capabilities=("bf16", "lora"),
        constraints={"minimum_runtime_version": "v1"},
    )


def _group(
    *,
    model_revision: str = "v1",
    tokenizer: str = "tokenizer-a",
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    target_class: str = "wsl2_rocm",
) -> CompatibilityGroup:
    target = _target(target_class)
    return CompatibilityGroup(
        group_id=f"group-{model_revision}-{target_class}",
        base_model_revision=_reference(
            "base_model_revision", "model-alpha", model_revision
        ),
        tokenizer_identity=_identity(tokenizer),
        rendering_template=_identity("render-template"),
        adapter_type="lora",
        target_modules=target_modules,
        runtime_targets=(
            RuntimeTargetConstraint(
                target_class,
                "rocm",
                target.runtime_contract,
                ("bf16", "lora"),
            ),
        ),
        merge_methods=("linear",),
    )


def _artifact(group: CompatibilityGroup) -> Artifact:
    return Artifact(
        artifact_id="artifact-alpha",
        project=_reference("project", "project-rewrite"),
        producing_run=_reference("run", "run-alpha"),
        adapter_type=group.adapter_type,
        content_kind=ArtifactContentKind.BUNDLE,
        content_identity=_identity("artifact-content"),
        base_model_revision=group.base_model_revision,
        tokenizer_identity=group.tokenizer_identity,
        compatibility_groups=(record_reference(group, group.group_id),),
        parent_artifacts=(),
        storage_references=(StorageReference("project_store", "artifact_alpha"),),
        integrity_evidence=_identity("artifact-integrity"),
        provenance=_identity("artifact-provenance"),
        lineage_evidence=_identity("artifact-lineage"),
    )


def _violation_codes(decision) -> set[str]:
    return {violation.code for violation in decision.violations}


def test_comparison_allows_different_models_under_same_task_policy() -> None:
    common = dict(
        task_definition=_reference("task_definition", "task-rewrite"),
        project_policy=_reference("project_policy", "policy-rewrite"),
        evaluation_policy=_identity("evaluation-policy"),
        objectives=("quality", "latency"),
    )
    left = ComparisonProfile(
        base_model_revision=_reference("base_model_revision", "model-a"), **common
    )
    right = ComparisonProfile(
        base_model_revision=_reference("base_model_revision", "model-b"), **common
    )

    assert check_comparison_compatibility(left, right).compatible
    assert not check_comparison_compatibility(
        left, replace(right, evaluation_policy=_identity("other-policy"))
    ).compatible
    assert "project_policy_mismatch" in _violation_codes(
        check_comparison_compatibility(
            left,
            replace(
                right,
                project_policy=_reference("project_policy", "policy-rewrite", "v2"),
            ),
        )
    )


@pytest.mark.parametrize(
    ("right", "expected_code"),
    [
        (_group(model_revision="v2"), "base_model_mismatch"),
        (_group(tokenizer="tokenizer-b"), "tokenizer_mismatch"),
        (_group(target_modules=("k_proj", "q_proj")), "target_modules_mismatch"),
    ],
)
def test_merge_rejects_incompatible_exact_contracts(
    right: CompatibilityGroup, expected_code: str
) -> None:
    decision = check_merge_compatibility(
        _group(),
        right,
        "linear",
        left_integrity_verified=True,
        right_integrity_verified=True,
    )

    assert expected_code in _violation_codes(decision)
    with pytest.raises(CompatibilityError):
        decision.require()


def test_merge_requires_supported_method_and_explicit_integrity_evidence() -> None:
    unverified = check_merge_compatibility(_group(), _group(), "linear")
    verified = check_merge_compatibility(
        _group(),
        _group(),
        "linear",
        left_integrity_verified=True,
        right_integrity_verified=True,
    )
    unsupported = check_merge_compatibility(
        _group(),
        _group(),
        "concatenate",
        left_integrity_verified=True,
        right_integrity_verified=True,
    )

    assert not unverified.compatible
    assert verified.compatible
    assert "merge_method_unsupported" in _violation_codes(unsupported)


def test_resume_requires_available_checkpoint_and_exact_training_state() -> None:
    resolution = _reference("recipe_resolution", "resolution-v1")
    checkpoint = ResumeCheckpoint(
        _identity("experiment-manifest"),
        resolution,
        _identity("training-state"),
        _reference("execution_target", "target-wsl2-rocm"),
        True,
    )
    request = ResumeRequest(
        _identity("experiment-manifest"),
        resolution,
        _identity("training-state"),
        _reference("execution_target", "target-wsl2-rocm"),
    )
    assert check_resume_compatibility(checkpoint, request).compatible

    changed = replace(
        request,
        training_state_identity=_identity("changed-training-state"),
        execution_target=_reference(
            "execution_target", "target-wsl2-rocm", "runtime-v2"
        ),
    )
    decision = check_resume_compatibility(replace(checkpoint, available=False), changed)
    assert _violation_codes(decision) >= {
        "checkpoint_unavailable",
        "training_state_mismatch",
        "execution_target_mismatch",
    }


def test_runtime_target_classes_are_explicit_and_never_silently_switched() -> None:
    group = _group(target_class="wsl2_rocm")
    artifact = _artifact(group)
    wsl = check_deployment_compatibility(
        artifact,
        group,
        _target("wsl2_rocm"),
        integrity_evidence=artifact.integrity_evidence,
    )
    windows = check_deployment_compatibility(
        artifact,
        group,
        _target("native_windows_rocm"),
        integrity_evidence=artifact.integrity_evidence,
    )

    assert wsl.compatible
    assert "runtime_target_undeclared" in _violation_codes(windows)
    assert "ready" not in wsl.to_dict()

    unrelated = _group(model_revision="v2")
    mismatch = check_deployment_compatibility(
        artifact,
        unrelated,
        _target("wsl2_rocm"),
        integrity_evidence=_identity("wrong-integrity"),
    )
    assert _violation_codes(mismatch) >= {
        "compatibility_group_mismatch",
        "base_model_mismatch",
        "integrity_evidence_mismatch",
    }
