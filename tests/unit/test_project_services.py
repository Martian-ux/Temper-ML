from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.projects import ProjectCreateRequest, ProjectService
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.policies import BaselinePolicy, PerModelBaseline
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import record_reference
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.store.evidence import TypedEvidenceStore


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _request() -> ProjectCreateRequest:
    task = TaskDefinition(
        task_id="task-slice-three",
        display_name="Synthetic rewrite",
        description="Rewrite synthetic text without changing named entities.",
        input_schema={"required": ["input"]},
        output_schema={"required": ["output"]},
        rendering_contract=_identity("renderer"),
        objectives=("preserve_entities",),
        capabilities=("text_generation",),
    )
    model = BaseModelRevision(
        model_id="model-synthetic",
        display_name="Synthetic model",
        model_family="fixture-family",
        architecture="fixture-causal-lm",
        source="public-fixture",
        revision="revision-one",
        weights_identity=_identity("weights"),
        tokenizer_identity=_identity("tokenizer"),
        license="Apache-2.0",
    )
    project = Project(
        project_id="project-slice-three",
        display_name="Slice three project",
        purpose="Exercise task-centered project services with synthetic records.",
        task_definition=record_reference(task),
        base_model_revisions=(record_reference(model),),
    )
    baseline = BaselinePolicy(
        "baseline-synthetic", (PerModelBaseline(_identity("comparison")),)
    )
    policy = ProjectPolicy(
        policy_id="policy-synthetic",
        project=record_reference(project),
        task_definition=record_reference(task),
        rendering_contract=task.rendering_contract,
        evaluation_policy=_identity("evaluation"),
        case_suites=(_identity("regression"),),
        readiness_policy=_identity("readiness"),
        retention_policy=_identity("retention"),
        approved_recipe_families=("balanced",),
        baseline_policy=record_reference(baseline),
        recommendation_policy=_identity("recommendation"),
    )
    return ProjectCreateRequest(task, project, baseline, policy, (model,))


def test_project_create_is_idempotent_and_recovers_an_interrupted_prefix(
    tmp_path: Path,
) -> None:
    request = _request()
    TypedEvidenceStore(tmp_path).write_record(request.task_definition)
    service = ProjectService(tmp_path)

    first = service.create(request)
    second = service.create(request)

    assert first.project.identity == second.project.identity
    assert first.project_policy.identity == second.project_policy.identity
    assert first.verification.record_count == 5
    assert first.verification.event_count == 1
    view = first.to_view()
    assert view["status"] == "open"
    assert view["base_model_revision_count"] == 1
    assert str(tmp_path) not in json.dumps(view)


def test_project_create_rejects_a_conflicting_revision_without_ambiguity(
    tmp_path: Path,
) -> None:
    request = _request()
    service = ProjectService(tmp_path)
    service.create(request)
    changed_project = replace(
        request.project,
        purpose="A different immutable purpose for the same logical project.",
    )
    changed_policy = replace(
        request.project_policy, project=record_reference(changed_project)
    )

    with pytest.raises(ApplicationServiceError, match="project_revision_conflict"):
        service.create(
            replace(
                request,
                project=changed_project,
                project_policy=changed_policy,
            )
        )

    projects = [
        stored
        for stored in service.store.iter_records()
        if stored.envelope.record_type == "project"
    ]
    assert len(projects) == 1
    assert projects[0].envelope.identity == request.project.identity


def test_project_open_requires_an_exact_policy_when_revisions_are_ambiguous(
    tmp_path: Path,
) -> None:
    request = _request()
    service = ProjectService(tmp_path)
    service.create(request)
    second_policy = replace(request.project_policy, policy_id="policy-synthetic-two")
    service.store.write_record(second_policy)

    with pytest.raises(ApplicationServiceError, match="project_policy_ambiguous"):
        service.open(request.project.project_id)

    opened = service.open(
        request.project.project_id,
        policy_id=request.project_policy.policy_id,
        policy_identity=request.project_policy.identity,
    )
    assert opened.project_policy.identity == request.project_policy.identity
