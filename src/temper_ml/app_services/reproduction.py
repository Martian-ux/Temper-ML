"""Execute one exact strict or explicitly adapted replay plan."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.experiments import ReplayMode, ReplayPlan
from temper_ml.app_services.runs import (
    RunExecutionResult,
    RunLaunchRequest,
    RunService,
)
from temper_ml.domain.experiments import ReproductionMode
from temper_ml.domain.records import record_reference
from temper_ml.runtime.fixture_adapter import FixtureAdapter, FixtureControl
from temper_ml.runtime.preflight import PreflightError, preflight
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest


@dataclass(frozen=True)
class ReplayExecutionRequest:
    """A ready plan and the complete new-run request that must match it."""

    plan: ReplayPlan
    launch: RunLaunchRequest

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ReplayPlan) or not isinstance(
            self.launch, RunLaunchRequest
        ):
            raise ApplicationServiceError("replay_execution_request_invalid")


@dataclass(frozen=True)
class ReplayExecutionResult:
    """One replay plan bound to its newly executed run evidence."""

    plan: ReplayPlan
    run: RunExecutionResult

    def to_view(self) -> dict[str, object]:
        return {
            "schema_version": "v1",
            "status": self.run.status.value,
            "mode": self.plan.mode.value,
            "exact_reproduction": self.plan.mode is ReplayMode.STRICT,
            "adapted_reproduction": self.plan.mode is ReplayMode.ADAPTED,
            "plan": self.plan.to_view(),
            "execution": self.run.to_view(),
        }


class ReproductionService:
    """Validate replay identity and delegate execution to the ordinary run service."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        adapter: FixtureAdapter | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.store = TypedEvidenceStore(self.project_root)
        self.adapter = adapter

    def execute(
        self,
        request: ReplayExecutionRequest,
        *,
        control: FixtureControl | None = None,
    ) -> ReplayExecutionResult:
        """Create a new run only when every launch input matches the ready plan."""

        if not isinstance(request, ReplayExecutionRequest):
            raise ApplicationServiceError("replay_execution_request_invalid")
        plan = request.plan
        launch = request.launch
        if not plan.ready:
            raise ApplicationServiceError("replay_plan_blocked")
        if launch.experiment != plan.planned_experiment:
            raise ApplicationServiceError("replay_experiment_mismatch")
        if (
            launch.prepared_dataset.version.identity
            != launch.experiment.dataset_version
        ):
            raise ApplicationServiceError("replay_dataset_mismatch")
        try:
            exact_preflight = preflight(
                launch.recipe_resolution,
                launch.hardware_requirements,
                launch.execution_target,
                launch.hardware_capability_profile,
                launch.estimate,
            )
        except PreflightError as exc:
            raise ApplicationServiceError(exc.code) from None
        if exact_preflight != plan.preflight:
            raise ApplicationServiceError("replay_preflight_mismatch")
        if plan.mode is ReplayMode.STRICT:
            if (
                plan.derivation is not None
                or plan.source_experiment != plan.planned_experiment
                or plan.source_experiment.manifest_identity
                != plan.planned_experiment.manifest_identity
            ):
                raise ApplicationServiceError("strict_replay_manifest_changed")
        elif plan.mode is ReplayMode.ADAPTED:
            derivation = plan.derivation
            if (
                derivation is None
                or derivation.reproduction_mode
                is not ReproductionMode.ADAPTED_REPRODUCTION
                or derivation.parent_experiment != plan.source_experiment
                or derivation.derived_experiment != plan.planned_experiment
                or not derivation.manifest_diff.changes
            ):
                raise ApplicationServiceError("adapted_replay_lineage_invalid")
            try:
                stored = self.store.read_record(record_reference(derivation)).record
            except EvidenceError:
                raise ApplicationServiceError(
                    "adapted_replay_lineage_missing"
                ) from None
            if stored != derivation:
                raise ApplicationServiceError("adapted_replay_lineage_mismatch")
        else:  # pragma: no cover - enum construction already prevents this
            raise ApplicationServiceError("replay_mode_invalid")
        execution_key = hashlib.sha256(
            f"{plan.plan_id}:{launch.run_id}".encode("utf-8")
        ).hexdigest()[:24]
        stream_id = f"replay-{execution_key}"
        event_prefix = stream_id
        try:
            self.store.append_event(
                stream_id,
                EventRequest(
                    f"{event_prefix}-started",
                    "replay_execution_started",
                    {
                        "mode": plan.mode.value,
                        "run_id": launch.run_id,
                        "source_manifest_identity": {
                            "algorithm": (
                                plan.source_experiment.manifest_identity.algorithm
                            ),
                            "value": plan.source_experiment.manifest_identity.value,
                        },
                        "planned_manifest_identity": {
                            "algorithm": (
                                plan.planned_experiment.manifest_identity.algorithm
                            ),
                            "value": plan.planned_experiment.manifest_identity.value,
                        },
                    },
                ),
            )
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        run_service = RunService(self.project_root, adapter=self.adapter)
        try:
            result = run_service.launch(launch, control=control)
        except ApplicationServiceError as exc:
            try:
                self.store.append_event(
                    stream_id,
                    EventRequest(
                        f"{event_prefix}-failed",
                        "replay_execution_failed",
                        {"failure_code": exc.code},
                    ),
                )
            except EvidenceError:
                pass
            raise
        try:
            self.store.append_event(
                stream_id,
                EventRequest(
                    f"{event_prefix}-completed",
                    "replay_execution_completed",
                    {
                        "mode": plan.mode.value,
                        "run_id": result.run.run_id,
                        "exact_reproduction": plan.mode is ReplayMode.STRICT,
                        "adapted_reproduction": plan.mode is ReplayMode.ADAPTED,
                    },
                ),
            )
            self.store.verify()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        return ReplayExecutionResult(plan, result)
