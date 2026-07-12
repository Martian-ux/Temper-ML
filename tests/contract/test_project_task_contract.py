import hashlib

import pytest

from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.policies import (
    BaselinePolicy,
    FixedReferenceBaseline,
    PerModelBaseline,
    ProjectChampionBaseline,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.records import (
    RecordReference,
    RecordValidationError,
    record_reference,
)
from temper_ml.domain.tasks import TaskDefinition


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _reference(kind: str, logical_id: str, revision: str = "v1") -> RecordReference:
    return RecordReference(
        kind, logical_id, _identity(f"{kind}:{logical_id}:{revision}")
    )


def _task() -> TaskDefinition:
    return TaskDefinition(
        task_id="task-rewrite",
        display_name="Synthetic rewrite",
        description="A bounded synthetic rewrite task.",
        input_schema={"required": ["input"]},
        output_schema={"required": ["output"]},
        rendering_contract=_identity("renderer"),
        objectives=("preserve_entities", "match_style"),
        capabilities=("text_generation",),
    )


def test_project_and_policy_pin_task_and_all_policy_dependencies() -> None:
    task = _task()
    task_ref = record_reference(task, task.task_id)
    project = Project(
        project_id="project-rewrite",
        display_name="Rewrite adapter",
        purpose="Train one adapter for the synthetic rewrite task.",
        task_definition=task_ref,
        base_model_revisions=(_reference("base_model_revision", "model-alpha"),),
    )
    baseline = BaselinePolicy(
        "baseline-default",
        (
            FixedReferenceBaseline(
                _identity("compare"), _reference("artifact", "artifact-reference")
            ),
            PerModelBaseline(_identity("compare")),
            ProjectChampionBaseline(_identity("compare")),
        ),
    )
    policy = ProjectPolicy(
        policy_id="policy-v1",
        project=record_reference(project, project.project_id),
        task_definition=task_ref,
        rendering_contract=task.rendering_contract,
        evaluation_policy=_identity("evaluation-policy"),
        case_suites=(_identity("regression-suite"), _identity("confirmation-suite")),
        readiness_policy=_identity("readiness-policy"),
        retention_policy=_identity("retention-policy"),
        approved_recipe_families=("balanced", "memory_saver"),
        baseline_policy=record_reference(baseline, baseline.policy_id),
        recommendation_policy=_identity("recommendation-policy"),
    )

    assert policy.to_payload()["task_definition"] == task_ref.to_dict()
    assert [rule["kind"] for rule in baseline.to_payload()["rules"]] == [
        "fixed_reference",
        "per_model",
        "project_champion",
    ]
    assert policy.identity != project.identity


def test_baseline_kinds_are_independent_and_fixed_reference_is_pinned() -> None:
    comparison = _identity("comparison")
    with pytest.raises(RecordValidationError, match="unsupported baseline"):
        BaselinePolicy("baseline-invalid", (object(),))

    with pytest.raises(RecordValidationError, match="pinned"):
        FixedReferenceBaseline(comparison, None)

    with pytest.raises(RecordValidationError, match="unique"):
        BaselinePolicy(
            "baseline-duplicate",
            (PerModelBaseline(comparison), PerModelBaseline(comparison)),
        )


def test_exact_base_model_revision_identity_does_not_follow_friendly_name() -> None:
    common = dict(
        model_id="model-alpha",
        display_name="Friendly Alpha",
        model_family="synthetic-family",
        architecture="synthetic-causal-lm",
        source="public-fixture",
        weights_identity=_identity("weights"),
        tokenizer_identity=_identity("tokenizer"),
        license="Apache-2.0",
    )
    first = BaseModelRevision(revision="revision-a", **common)
    second = BaseModelRevision(revision="revision-b", **common)

    assert first.display_name == second.display_name
    assert first.identity != second.identity


def test_reference_sets_deduplicate_by_immutable_identity_not_alias() -> None:
    task = _task()
    model = _reference("base_model_revision", "model-alpha")
    alias = RecordReference("base_model_revision", "model-alias", model.identity)

    with pytest.raises(RecordValidationError, match="duplicates"):
        Project(
            project_id="project-alias-test",
            display_name="Alias test",
            purpose="Reject two logical aliases for one immutable model revision.",
            task_definition=record_reference(task, task.task_id),
            base_model_revisions=(model, alias),
        )
