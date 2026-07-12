from dataclasses import replace
import hashlib

import pytest

from temper_ml.domain.experiments import (
    DiffOperation,
    Experiment,
    ExperimentDerivation,
    ManifestChange,
    ManifestDiff,
    ReproductionMode,
    derive_experiment,
    manifest_identity,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    record_reference,
)
from temper_ml.domain.runs import Run


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(kind: str, logical_id: str, revision: str = "v1") -> RecordReference:
    return RecordReference(
        kind, logical_id, _identity(f"{kind}:{logical_id}:{revision}")
    )


def _experiment(
    experiment_id: str = "experiment-parent", resolution_revision: str = "v1"
) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        project=_reference("project", "project-rewrite"),
        project_policy=_reference("project_policy", "policy-v1"),
        task_definition=_reference("task_definition", "task-rewrite"),
        dataset_version=_identity("dataset-version"),
        base_model_revision=_reference("base_model_revision", "model-alpha"),
        tokenizer_identity=_identity("tokenizer"),
        recipe=_reference("recipe", "recipe-balanced"),
        recipe_resolution=_reference(
            "recipe_resolution", "resolution-balanced", resolution_revision
        ),
        evaluation_policy=_identity("evaluation-policy"),
        compatibility_group=_reference("compatibility_group", "group-alpha"),
        hardware_requirements=_reference(
            "hardware_requirements", "requirements-balanced"
        ),
        execution_target=_reference("execution_target", "target-wsl2-rocm"),
    )


def test_derived_experiment_has_new_identity_reason_and_replayable_exact_diff() -> None:
    parent = _experiment()
    derived = _experiment("experiment-derived", "v2")

    evidence = derive_experiment(
        parent,
        derived,
        derivation_id="derivation-resolution-change",
        diff_id="diff-resolution-change",
        reason_code="hardware_adaptation",
        reason="The original resolved manifest exceeded the declared memory budget.",
        reproduction_mode=ReproductionMode.ADAPTED_REPRODUCTION,
    )

    assert parent.identity != derived.identity
    assert parent.manifest_identity != derived.manifest_identity
    assert evidence.reason_code == "hardware_adaptation"
    assert evidence.reproduction_mode is ReproductionMode.ADAPTED_REPRODUCTION
    assert evidence.manifest_diff.apply(parent.scientific_manifest()) == (
        derived.scientific_manifest()
    )
    assert [change.path for change in evidence.manifest_diff.changes] == [
        "/recipe_resolution/identity/value"
    ]
    assert evidence.parent_reference.identity == parent.identity
    assert evidence.derived_reference.identity == derived.identity


def test_derivation_constructor_binds_diff_to_embedded_experiments() -> None:
    parent = _experiment()
    derived = _experiment("experiment-derived", "v2")
    diff = ManifestDiff.between(
        "diff-resolution", parent.scientific_manifest(), derived.scientific_manifest()
    )
    unrelated_parent = replace(parent, dataset_version=_identity("other-dataset"))

    with pytest.raises(RecordValidationError, match="do not match"):
        ExperimentDerivation(
            derivation_id="derivation-invalid",
            parent_experiment=unrelated_parent,
            derived_experiment=derived,
            reproduction_mode=ReproductionMode.SCIENTIFIC_DERIVATION,
            reason_code="resolution_change",
            reason="Synthetic invalid lineage claim.",
            manifest_diff=diff,
        )


def test_new_logical_id_without_material_manifest_change_is_not_a_derivation() -> None:
    parent = _experiment()
    renamed_only = replace(parent, experiment_id="experiment-renamed")

    with pytest.raises(RecordValidationError, match="no-op"):
        derive_experiment(
            parent,
            renamed_only,
            derivation_id="derivation-noop",
            diff_id="diff-noop",
            reason_code="rename_only",
            reason="Only the logical name changed.",
        )


def test_manifest_diff_distinguishes_missing_from_null_and_escapes_pointers() -> None:
    parent = {
        "a/b": None,
        "x~y": {"keep": 1},
        "removed": 0,
        "array": [1, 2],
    }
    derived = {
        "a/b": "now-present",
        "x~y": {"keep": 1, "new": None},
        "array": [1, 3],
    }
    diff = ManifestDiff.between("diff-pointer-cases", parent, derived)

    assert [change.path for change in diff.changes] == [
        "/array",
        "/a~1b",
        "/removed",
        "/x~0y/new",
    ]
    assert diff.changes[-1].operation is DiffOperation.ADD
    assert diff.changes[-1].to_dict()["after"] is None
    assert diff.apply(parent) == derived
    assert parent["array"] == [1, 2]


def test_manifest_diff_rejects_overlap_and_wrong_parent_state() -> None:
    parent = {"a": {"b": 1}}
    derived = {"a": {"b": 2}}
    with pytest.raises(RecordValidationError, match="overlap"):
        ManifestDiff(
            "diff-overlap",
            manifest_identity(parent),
            manifest_identity(derived),
            (
                ManifestChange("/a", DiffOperation.REPLACE, {"b": 1}, {"b": 2}),
                ManifestChange("/a/b", DiffOperation.REPLACE, 1, 2),
            ),
        )

    diff = ManifestDiff.between("diff-state", parent, derived)
    with pytest.raises(RecordValidationError, match="identity mismatch"):
        diff.apply({"a": {"b": 99}})

    with pytest.raises(RecordValidationError, match="root can only be replaced"):
        ManifestChange("", DiffOperation.ADD, after={"new": True})
    with pytest.raises(RecordValidationError, match="root can only be replaced"):
        ManifestChange("", DiffOperation.REMOVE, before={"old": True})


def test_retries_are_distinct_runs_under_one_unchanged_experiment() -> None:
    experiment = _experiment()
    experiment_ref = record_reference(experiment, experiment.experiment_id)
    common = dict(
        experiment=experiment_ref,
        experiment_manifest_identity=experiment.manifest_identity,
        hardware_capability_profile=_reference(
            "hardware_capability_profile", "profile-synthetic"
        ),
        execution_target=_reference("execution_target", "target-wsl2-rocm"),
        runtime_identity=_identity("runtime"),
        request_identity=_identity("request"),
        training_state_identity=_identity("training-state"),
    )
    first = Run(run_id="run-attempt-1", attempt_number=1, **common)
    retry = Run(
        run_id="run-attempt-2",
        attempt_number=2,
        retry_of=record_reference(first, first.run_id),
        **common,
    )

    assert first.experiment == retry.experiment
    assert first.identity != retry.identity
    assert "status" not in first.to_payload()
