"""Task-centered project creation, recovery, and opening services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from temper_ml.app_services._records import (
    require_no_conflicting_logical_revision,
    write_record_idempotently,
)
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.domain.base_models import BaseModelRevision
from temper_ml.domain.policies import BaselinePolicy
from temper_ml.domain.projects import Project, ProjectPolicy
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import record_reference
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.store.evidence import ProjectVerification, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest


@dataclass(frozen=True)
class ProjectCreateRequest:
    """All immutable records needed to establish one usable project revision."""

    task_definition: TaskDefinition
    project: Project
    baseline_policy: BaselinePolicy
    project_policy: ProjectPolicy
    base_model_revisions: tuple[BaseModelRevision, ...] = ()


@dataclass(frozen=True)
class OpenedProject:
    """One exact, verified project and its governing task and policy."""

    project: Project
    task_definition: TaskDefinition
    project_policy: ProjectPolicy
    verification: ProjectVerification

    def to_view(self) -> dict[str, object]:
        return {
            "status": "open",
            "project": record_reference(self.project).to_dict(),
            "project_policy": record_reference(self.project_policy).to_dict(),
            "task_definition": record_reference(self.task_definition).to_dict(),
            "display_name": self.project.display_name,
            "purpose": self.project.purpose,
            "base_model_revision_count": len(self.project.base_model_revisions),
            "store": self.verification.to_dict(),
        }


class ProjectService:
    """Create or open task-centered projects over the canonical evidence store."""

    def __init__(self, project_root: Path | str) -> None:
        self.store = TypedEvidenceStore(project_root)

    def create(self, request: ProjectCreateRequest) -> OpenedProject:
        """Create a project idempotently, including interrupted-write recovery."""

        self._validate_create_request(request)
        records = (
            request.task_definition,
            *request.base_model_revisions,
            request.project,
            request.baseline_policy,
            request.project_policy,
        )
        for record in records:
            require_no_conflicting_logical_revision(
                self.store,
                record,
                conflict_code="project_revision_conflict",
            )
        for record in records:
            write_record_idempotently(
                self.store,
                record,
                conflict_code="project_revision_conflict",
            )
        self.store.append_event(
            "project-lifecycle",
            EventRequest(
                f"project-created-{request.project.project_id}",
                "project_created",
                {
                    "project_revision_created": True,
                    "policy_revision_created": True,
                    "base_model_revision_count": len(request.base_model_revisions),
                },
            ),
        )
        return self.open(
            request.project.project_id,
            project_identity=request.project.identity,
            policy_id=request.project_policy.policy_id,
            policy_identity=request.project_policy.identity,
        )

    def open(
        self,
        project_id: str,
        *,
        project_identity: ContentIdentity | None = None,
        policy_id: str | None = None,
        policy_identity: ContentIdentity | None = None,
    ) -> OpenedProject:
        """Open one unambiguous project revision after full store verification."""

        verification = self.store.verify()
        project_record = self.store.inspect_manifest(
            "project", project_id, project_identity
        ).to_record()
        if not isinstance(project_record, Project):
            raise ApplicationServiceError("project_record_invalid")

        policies = []
        for stored in self.store.iter_records():
            candidate = stored.record
            if not isinstance(candidate, ProjectPolicy):
                continue
            if candidate.project != record_reference(project_record):
                continue
            if policy_id is not None and candidate.policy_id != policy_id:
                continue
            if policy_identity is not None and candidate.identity != policy_identity:
                continue
            policies.append(candidate)
        if not policies:
            raise ApplicationServiceError("project_policy_not_found")
        if len(policies) != 1:
            raise ApplicationServiceError("project_policy_ambiguous")
        policy = policies[0]

        task_record = self.store.read_record(project_record.task_definition).record
        if not isinstance(task_record, TaskDefinition):
            raise ApplicationServiceError("project_task_invalid")
        self._validate_opened(project_record, task_record, policy)
        return OpenedProject(project_record, task_record, policy, verification)

    @staticmethod
    def _validate_create_request(request: ProjectCreateRequest) -> None:
        if not isinstance(request, ProjectCreateRequest):
            raise ApplicationServiceError("invalid_project_request")
        if not isinstance(request.task_definition, TaskDefinition) or not isinstance(
            request.project, Project
        ):
            raise ApplicationServiceError("invalid_project_request")
        if not isinstance(request.baseline_policy, BaselinePolicy) or not isinstance(
            request.project_policy, ProjectPolicy
        ):
            raise ApplicationServiceError("invalid_project_request")
        if not isinstance(request.base_model_revisions, tuple) or any(
            not isinstance(model, BaseModelRevision)
            for model in request.base_model_revisions
        ):
            raise ApplicationServiceError("invalid_project_request")
        if request.project.task_definition != record_reference(request.task_definition):
            raise ApplicationServiceError("project_task_mismatch")
        expected_models = tuple(
            sorted(
                (record_reference(model) for model in request.base_model_revisions),
                key=lambda reference: (
                    reference.identity.value,
                    reference.logical_id,
                ),
            )
        )
        if request.project.base_model_revisions != expected_models:
            raise ApplicationServiceError("project_models_mismatch")
        if request.project_policy.project != record_reference(request.project):
            raise ApplicationServiceError("project_policy_mismatch")
        if request.project_policy.task_definition != record_reference(
            request.task_definition
        ):
            raise ApplicationServiceError("project_policy_mismatch")
        if (
            request.project_policy.rendering_contract
            != request.task_definition.rendering_contract
        ):
            raise ApplicationServiceError("project_policy_rendering_mismatch")
        if request.project_policy.baseline_policy != record_reference(
            request.baseline_policy
        ):
            raise ApplicationServiceError("project_baseline_mismatch")

    @staticmethod
    def _validate_opened(
        project: Project,
        task_definition: TaskDefinition,
        project_policy: ProjectPolicy,
    ) -> None:
        if project.task_definition != record_reference(task_definition):
            raise ApplicationServiceError("project_task_mismatch")
        if project_policy.project != record_reference(project):
            raise ApplicationServiceError("project_policy_mismatch")
        if project_policy.task_definition != record_reference(task_definition):
            raise ApplicationServiceError("project_policy_mismatch")
        if project_policy.rendering_contract != task_definition.rendering_contract:
            raise ApplicationServiceError("project_policy_rendering_mismatch")
