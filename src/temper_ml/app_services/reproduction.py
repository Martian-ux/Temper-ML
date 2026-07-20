"""Execute and durably reconcile strict or explicitly adapted replay plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.experiments import ReplayMode, ReplayPlan
from temper_ml.app_services.runs import (
    RunExecutionResult,
    RunLaunchRequest,
    RunLifecycleStatus,
    RunService,
)
from temper_ml.domain.artifacts import Artifact
from temper_ml.domain.experiments import Experiment, ReproductionMode
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordValidationError,
    identity_fields,
    parse_identity,
    record_reference,
    require_identifier,
)
from temper_ml.domain.runs import ResolvedRuntimeRequest, Run
from temper_ml.runtime.fixture_adapter import FixtureAdapter, FixtureControl
from temper_ml.runtime.preflight import PreflightError, PreflightResult, preflight
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore
from temper_ml.store.event_stream import EventRequest, StoredEvent


_REPLAY_TERMINAL_EVENT_BY_STATUS = {
    RunLifecycleStatus.COMPLETED: "replay_execution_completed",
    RunLifecycleStatus.CANCELLED: "replay_execution_cancelled",
    RunLifecycleStatus.INTERRUPTED: "replay_execution_interrupted",
    RunLifecycleStatus.FAILED: "replay_execution_failed",
    RunLifecycleStatus.PREFLIGHT_BLOCKED: "replay_execution_failed",
}
_REPLAY_TERMINAL_EVENTS = frozenset(_REPLAY_TERMINAL_EVENT_BY_STATUS.values())


@dataclass(frozen=True)
class ReplayExecutionRequest:
    """A ready plan and the complete new-run request that must match it."""

    plan: ReplayPlan
    launch: RunLaunchRequest
    candidate_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ReplayPlan) or not isinstance(
            self.launch, RunLaunchRequest
        ):
            raise ApplicationServiceError("replay_execution_request_invalid")
        if self.candidate_key is not None:
            try:
                require_identifier("candidate_key", self.candidate_key)
            except RecordValidationError:
                raise ApplicationServiceError(
                    "replay_execution_request_invalid"
                ) from None


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
            "reconciled": False,
        }


@dataclass(frozen=True)
class ReplayReconciliationResult:
    """A replay terminal reconstructed only from durable canonical evidence."""

    plan_id: str
    mode: ReplayMode
    candidate_key: str | None
    run_id: str
    run_identity: ContentIdentity
    runtime_request_identity: ContentIdentity
    status: RunLifecycleStatus
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not self.status.terminal:
            raise ApplicationServiceError("replay_reconciliation_incomplete")
        if (self.status is RunLifecycleStatus.FAILED) != (
            self.failure_code is not None
        ):
            raise ApplicationServiceError("replay_reconciliation_invalid")

    def to_view(self) -> dict[str, object]:
        plan: dict[str, object] = {
            "plan_id": self.plan_id,
            "mode": self.mode.value,
            "status": "ready",
        }
        if self.candidate_key is not None:
            plan["candidate_key"] = self.candidate_key
        execution: dict[str, object] = {
            "status": self.status.value,
            "run_id": self.run_id,
            "run_identity": identity_fields(self.run_identity),
            "runtime_request_identity": identity_fields(self.runtime_request_identity),
            "reconciled": True,
        }
        if self.failure_code is not None:
            execution["failure_code"] = self.failure_code
        return {
            "schema_version": "v1",
            "status": self.status.value,
            "mode": self.mode.value,
            "exact_reproduction": self.mode is ReplayMode.STRICT,
            "adapted_reproduction": self.mode is ReplayMode.ADAPTED,
            "plan": plan,
            "execution": execution,
            "reconciled": True,
        }


@dataclass(frozen=True)
class _ReplayIntent:
    plan_id: str
    mode: ReplayMode
    candidate_key: str | None
    run_id: str
    run_identity: ContentIdentity
    request_id: str
    runtime_request_identity: ContentIdentity
    artifact_id: str
    source_experiment_id: str
    source_experiment_identity: ContentIdentity
    planned_experiment_id: str
    planned_experiment_identity: ContentIdentity
    source_manifest_identity: ContentIdentity
    planned_manifest_identity: ContentIdentity

    @property
    def stream_id(self) -> str:
        execution_key = hashlib.sha256(
            f"{self.plan_id}:{self.run_id}".encode("utf-8")
        ).hexdigest()[:24]
        return f"replay-{execution_key}"

    def to_payload(self) -> dict[str, object]:
        return {
            "intent_schema_version": "v2",
            "plan_id": self.plan_id,
            "mode": self.mode.value,
            "candidate_key": self.candidate_key,
            "run_id": self.run_id,
            "run_identity": identity_fields(self.run_identity),
            "request_id": self.request_id,
            "runtime_request_identity": identity_fields(self.runtime_request_identity),
            "artifact_id": self.artifact_id,
            "source_experiment_id": self.source_experiment_id,
            "source_experiment_identity": identity_fields(
                self.source_experiment_identity
            ),
            "planned_experiment_id": self.planned_experiment_id,
            "planned_experiment_identity": identity_fields(
                self.planned_experiment_identity
            ),
            "source_manifest_identity": identity_fields(self.source_manifest_identity),
            "planned_manifest_identity": identity_fields(
                self.planned_manifest_identity
            ),
        }

    def terminal_payload(
        self,
        status: RunLifecycleStatus,
        *,
        failure_code: str | None = None,
    ) -> dict[str, object]:
        return {
            "intent_schema_version": "v2",
            "plan_id": self.plan_id,
            "mode": self.mode.value,
            "candidate_key": self.candidate_key,
            "run_id": self.run_id,
            "run_identity": identity_fields(self.run_identity),
            "runtime_request_identity": identity_fields(self.runtime_request_identity),
            "run_status": status.value,
            "exact_reproduction": self.mode is ReplayMode.STRICT,
            "adapted_reproduction": self.mode is ReplayMode.ADAPTED,
            "failure_code": failure_code,
        }


class ReproductionService:
    """Validate replay identity, execute once, and reconcile from durable evidence."""

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
    ) -> ReplayExecutionResult | ReplayReconciliationResult:
        """Create a new run only when every launch input matches the ready plan."""

        if not isinstance(request, ReplayExecutionRequest):
            raise ApplicationServiceError("replay_execution_request_invalid")
        plan = request.plan
        launch = request.launch
        exact_preflight = self._validate_request(plan, launch)
        run_service = RunService(self.project_root, adapter=self.adapter)
        expected_request, expected_run = run_service.planned_first_attempt(
            launch, exact_preflight
        )
        intent = _ReplayIntent(
            plan_id=plan.plan_id,
            mode=plan.mode,
            candidate_key=request.candidate_key,
            run_id=launch.run_id,
            run_identity=expected_run.identity,
            request_id=launch.request_id,
            runtime_request_identity=expected_request.identity,
            artifact_id=launch.artifact_id,
            source_experiment_id=plan.source_experiment.experiment_id,
            source_experiment_identity=plan.source_experiment.identity,
            planned_experiment_id=plan.planned_experiment.experiment_id,
            planned_experiment_identity=plan.planned_experiment.identity,
            source_manifest_identity=plan.source_experiment.manifest_identity,
            planned_manifest_identity=plan.planned_experiment.manifest_identity,
        )
        events = self._replay_events(intent.stream_id)
        _, terminal = self._validate_replay_events(events, expected_intent=intent)
        if terminal is not None:
            return terminal
        if not events:
            self._append_replay_event(
                intent.stream_id,
                EventRequest(
                    f"{intent.stream_id}-started",
                    "replay_execution_started",
                    intent.to_payload(),
                ),
            )
        else:
            reconciled = self._reconcile_intent(intent, run_service)
            if reconciled is not None:
                self._append_terminal(intent, reconciled.status)
                return reconciled
            if self._run_status_or_none(run_service, intent.run_id) is not None:
                raise ApplicationServiceError("replay_reconciliation_required")

        try:
            result = run_service.launch(launch, control=control)
        except ApplicationServiceError as exc:
            reconciled = self._reconcile_intent(intent, run_service)
            if reconciled is not None:
                self._append_terminal(
                    intent,
                    reconciled.status,
                    failure_code=reconciled.failure_code,
                )
                if reconciled.status in {
                    RunLifecycleStatus.COMPLETED,
                    RunLifecycleStatus.CANCELLED,
                    RunLifecycleStatus.INTERRUPTED,
                }:
                    return reconciled
                raise
            try:
                self._append_terminal(
                    intent,
                    RunLifecycleStatus.FAILED,
                    failure_code=exc.code,
                )
            except ApplicationServiceError:
                pass
            raise
        if (
            result.run != expected_run
            or result.runtime_request != expected_request
            or result.run.run_id != launch.run_id
        ):
            raise ApplicationServiceError("replay_run_identity_mismatch")
        self._append_terminal(intent, result.status)
        return ReplayExecutionResult(plan, result)

    def reconcile_pending(self) -> tuple[ReplayReconciliationResult, ...]:
        """Close replay streams from durable run identities without heavy bytes."""

        try:
            self.store.verify()
            streams = tuple(
                snapshot
                for snapshot in self.store.iter_streams()
                if snapshot.stream_id.startswith("replay-")
            )
        except EvidenceError as exc:
            if exc.code in {
                "project_not_found",
                "store_missing",
                "store_not_found",
                "store_root_missing",
            }:
                return ()
            raise ApplicationServiceError(exc.code) from None
        run_service = RunService(self.project_root, adapter=self.adapter)
        results: list[ReplayReconciliationResult] = []
        for snapshot in streams:
            intent, terminal = self._validate_replay_events(snapshot.events)
            if intent is None:
                continue
            if snapshot.stream_id != intent.stream_id:
                raise ApplicationServiceError("replay_execution_evidence_conflict")
            if terminal is not None:
                results.append(terminal)
                continue
            reconciled = self._reconcile_intent(intent, run_service)
            if reconciled is None:
                continue
            self._append_terminal(
                intent,
                reconciled.status,
                failure_code=reconciled.failure_code,
            )
            results.append(reconciled)
        return tuple(results)

    def reconcile_plan(
        self,
        plan_id: str,
        *,
        candidate_key: str | None = None,
        mode: str | None = None,
    ) -> ReplayReconciliationResult:
        """Return one exact durable terminal after process-local draft loss."""

        try:
            require_identifier("plan_id", plan_id)
        except RecordValidationError:
            raise ApplicationServiceError("replay_plan_mismatch") from None
        results = self.reconcile_pending()
        matches = tuple(item for item in results if item.plan_id == plan_id)
        if len(matches) != 1:
            raise ApplicationServiceError("replay_plan_required")
        result = matches[0]
        if candidate_key is not None and candidate_key != result.candidate_key:
            raise ApplicationServiceError("replay_candidate_plan_mismatch")
        if mode is not None and mode != result.mode.value:
            raise ApplicationServiceError("replay_mode_plan_mismatch")
        return result

    def _validate_request(
        self, plan: ReplayPlan, launch: RunLaunchRequest
    ) -> PreflightResult:
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
        return exact_preflight

    def _reconcile_intent(
        self,
        intent: _ReplayIntent,
        run_service: RunService,
    ) -> ReplayReconciliationResult | None:
        try:
            records = tuple(item.record for item in self.store.iter_records())
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        runs = tuple(
            item
            for item in records
            if isinstance(item, Run) and item.run_id == intent.run_id
        )
        requests = tuple(
            item
            for item in records
            if isinstance(item, ResolvedRuntimeRequest)
            and item.identity == intent.runtime_request_identity
        )
        source_experiments = tuple(
            item
            for item in records
            if isinstance(item, Experiment)
            and item.experiment_id == intent.source_experiment_id
            and item.identity == intent.source_experiment_identity
        )
        planned_experiments = tuple(
            item
            for item in records
            if isinstance(item, Experiment)
            and item.experiment_id == intent.planned_experiment_id
            and item.identity == intent.planned_experiment_identity
        )
        if not runs and not requests:
            return None
        if (
            len(runs) != 1
            or len(requests) != 1
            or len(source_experiments) != 1
            or len(planned_experiments) != 1
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        run = runs[0]
        runtime_request = requests[0]
        source_experiment = source_experiments[0]
        planned_experiment = planned_experiments[0]
        if (
            source_experiment.manifest_identity != intent.source_manifest_identity
            or planned_experiment.manifest_identity != intent.planned_manifest_identity
            or run.identity != intent.run_identity
            or run.request_identity != intent.runtime_request_identity
            or runtime_request.request_id != intent.request_id
            or run.experiment.logical_id != intent.planned_experiment_id
            or run.experiment.identity != intent.planned_experiment_identity
            or runtime_request.experiment != run.experiment
            or run.experiment_manifest_identity != intent.planned_manifest_identity
            or runtime_request.experiment_manifest_identity
            != intent.planned_manifest_identity
            or record_reference(planned_experiment) != run.experiment
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        status = self._run_status_or_none(run_service, intent.run_id)
        if status is None or status is RunLifecycleStatus.RUNNING:
            return None
        events = run_service._events(intent.run_id)
        launched = tuple(
            event for event in events if event.event_type == "run_launched"
        )
        if (
            len(launched) != 1
            or _payload_identity(launched[0].payload, "run_identity")
            != intent.run_identity
            or _payload_identity(launched[0].payload, "runtime_request_identity")
            != intent.runtime_request_identity
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        expected_terminal = {
            RunLifecycleStatus.PREFLIGHT_BLOCKED: "run_preflight_blocked",
            RunLifecycleStatus.CANCELLED: "run_cancelled",
            RunLifecycleStatus.INTERRUPTED: "run_interrupted",
            RunLifecycleStatus.COMPLETED: "run_completed",
            RunLifecycleStatus.FAILED: "run_failed",
        }[status]
        terminals = tuple(
            event for event in events if event.event_type == expected_terminal
        )
        if len(terminals) != 1:
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        terminal = terminals[0]
        failure_code: str | None = None
        if status is RunLifecycleStatus.FAILED:
            raw_failure_code = terminal.payload.get("failure_code")
            if not isinstance(raw_failure_code, str) or not raw_failure_code:
                raise ApplicationServiceError("replay_execution_evidence_conflict")
            failure_code = raw_failure_code
        if status is RunLifecycleStatus.COMPLETED:
            artifacts = tuple(
                item
                for item in records
                if isinstance(item, Artifact)
                and (
                    item.artifact_id == intent.artifact_id
                    or item.producing_run == record_reference(run)
                )
            )
            if (
                len(artifacts) != 1
                or artifacts[0].artifact_id != intent.artifact_id
                or artifacts[0].producing_run != record_reference(run)
                or _payload_identity(terminal.payload, "artifact_identity")
                != artifacts[0].identity
            ):
                raise ApplicationServiceError("replay_execution_evidence_conflict")
        return ReplayReconciliationResult(
            intent.plan_id,
            intent.mode,
            intent.candidate_key,
            intent.run_id,
            intent.run_identity,
            intent.runtime_request_identity,
            status,
            failure_code,
        )

    def _run_status_or_none(
        self, run_service: RunService, run_id: str
    ) -> RunLifecycleStatus | None:
        try:
            return run_service.status(run_id)
        except ApplicationServiceError as exc:
            if exc.code == "run_not_found":
                return None
            raise

    def _replay_events(self, stream_id: str) -> tuple[StoredEvent, ...]:
        try:
            return next(
                (
                    snapshot.events
                    for snapshot in self.store.iter_streams()
                    if snapshot.stream_id == stream_id
                ),
                (),
            )
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None

    def _validate_replay_events(
        self,
        events: tuple[StoredEvent, ...],
        *,
        expected_intent: _ReplayIntent | None = None,
    ) -> tuple[_ReplayIntent | None, ReplayReconciliationResult | None]:
        if not events:
            return expected_intent, None
        event_types = tuple(event.event_type for event in events)
        if (
            event_types[0] != "replay_execution_started"
            or event_types.count("replay_execution_started") != 1
            or any(
                event_type not in {"replay_execution_started", *_REPLAY_TERMINAL_EVENTS}
                for event_type in event_types
            )
            or sum(event_type in _REPLAY_TERMINAL_EVENTS for event_type in event_types)
            > 1
            or len(events) > 2
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        intent = _intent_from_payload(events[0].payload)
        if expected_intent is not None and intent != expected_intent:
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        if events[0].idempotency_key != f"{intent.stream_id}-started":
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        if len(events) == 1:
            return intent, None
        terminal = events[1]
        try:
            raw_status = terminal.payload.get("run_status")
            if not isinstance(raw_status, str):
                raise ValueError
            status = RunLifecycleStatus(raw_status)
        except (TypeError, ValueError):
            raise ApplicationServiceError(
                "replay_execution_evidence_conflict"
            ) from None
        expected_event_type = _REPLAY_TERMINAL_EVENT_BY_STATUS.get(status)
        if expected_event_type != terminal.event_type:
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        expected_key = (
            f"{intent.stream_id}-"
            f"{terminal.event_type.removeprefix('replay_execution_')}"
        )
        if terminal.idempotency_key != expected_key:
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        failure_code = terminal.payload.get("failure_code")
        if failure_code is not None and not isinstance(failure_code, str):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        if status is RunLifecycleStatus.FAILED:
            if not isinstance(failure_code, str) or not failure_code:
                raise ApplicationServiceError("replay_execution_evidence_conflict")
        elif failure_code is not None:
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        if dict(terminal.payload) != intent.terminal_payload(
            status, failure_code=failure_code
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        return intent, ReplayReconciliationResult(
            intent.plan_id,
            intent.mode,
            intent.candidate_key,
            intent.run_id,
            intent.run_identity,
            intent.runtime_request_identity,
            status,
            failure_code,
        )

    def _append_replay_event(self, stream_id: str, request: EventRequest) -> None:
        try:
            self.store.append_event(stream_id, request)
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None

    def _append_terminal(
        self,
        intent: _ReplayIntent,
        status: RunLifecycleStatus,
        *,
        failure_code: str | None = None,
    ) -> None:
        event_type = _REPLAY_TERMINAL_EVENT_BY_STATUS.get(status)
        if event_type is None:
            raise ApplicationServiceError("replay_terminal_status_invalid")
        suffix = event_type.removeprefix("replay_execution_")
        self._append_replay_event(
            intent.stream_id,
            EventRequest(
                f"{intent.stream_id}-{suffix}",
                event_type,
                intent.terminal_payload(status, failure_code=failure_code),
            ),
        )
        try:
            self.store.verify()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None


def _intent_from_payload(payload: Mapping[str, object]) -> _ReplayIntent:
    required = {
        "intent_schema_version",
        "plan_id",
        "mode",
        "candidate_key",
        "run_id",
        "run_identity",
        "request_id",
        "runtime_request_identity",
        "artifact_id",
        "source_experiment_id",
        "source_experiment_identity",
        "planned_experiment_id",
        "planned_experiment_identity",
        "source_manifest_identity",
        "planned_manifest_identity",
    }
    if set(payload) != required or payload.get("intent_schema_version") != "v2":
        raise ApplicationServiceError("replay_execution_evidence_conflict")
    try:
        mode = ReplayMode(payload.get("mode"))
        candidate_key = payload.get("candidate_key")
        if candidate_key is not None:
            candidate_key = _required_identifier(payload, "candidate_key")
        intent = _ReplayIntent(
            plan_id=_required_identifier(payload, "plan_id"),
            mode=mode,
            candidate_key=candidate_key,
            run_id=_required_identifier(payload, "run_id"),
            run_identity=_payload_identity(payload, "run_identity"),
            request_id=_required_identifier(payload, "request_id"),
            runtime_request_identity=_payload_identity(
                payload, "runtime_request_identity"
            ),
            artifact_id=_required_identifier(payload, "artifact_id"),
            source_experiment_id=_required_identifier(payload, "source_experiment_id"),
            source_experiment_identity=_payload_identity(
                payload, "source_experiment_identity"
            ),
            planned_experiment_id=_required_identifier(
                payload, "planned_experiment_id"
            ),
            planned_experiment_identity=_payload_identity(
                payload, "planned_experiment_identity"
            ),
            source_manifest_identity=_payload_identity(
                payload, "source_manifest_identity"
            ),
            planned_manifest_identity=_payload_identity(
                payload, "planned_manifest_identity"
            ),
        )
        source_identity = (
            intent.source_experiment_id,
            intent.source_experiment_identity,
            intent.source_manifest_identity,
        )
        planned_identity = (
            intent.planned_experiment_id,
            intent.planned_experiment_identity,
            intent.planned_manifest_identity,
        )
        if (
            intent.mode is ReplayMode.STRICT and source_identity != planned_identity
        ) or (
            intent.mode is ReplayMode.ADAPTED and source_identity == planned_identity
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        return intent
    except (ApplicationServiceError, RecordValidationError, TypeError, ValueError):
        raise ApplicationServiceError("replay_execution_evidence_conflict") from None


def _required_identifier(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise RecordValidationError(f"{field} is invalid")
    return require_identifier(field, value)


def _payload_identity(payload: Mapping[str, object], field: str) -> ContentIdentity:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{field} is invalid")
    return parse_identity(value, field=field)
