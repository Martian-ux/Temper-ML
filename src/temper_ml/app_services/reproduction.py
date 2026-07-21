"""Execute and durably reconcile strict or explicitly adapted replay plans."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace as dataclass_replace
import errno
import hashlib
import importlib
import os
from pathlib import Path
import stat
import time
from typing import Any

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.experiments import ReplayMode, ReplayPlan
from temper_ml.app_services.runs import (
    RUNTIME_OUTPUT_DIRECTORY,
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
from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    ensure_safe_directory,
    is_link_or_reparse,
    require_safe_regular_file,
    safe_path_stat,
    same_file_object,
)
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
_REPLAY_PLANNING_WAIT_SECONDS = 5.0
_REPLAY_PLANNING_RETRY_SECONDS = 0.01


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
            "run_id": self.run_id,
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
    run_ownership_identity: ContentIdentity | None
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
        payload: dict[str, object] = {
            "intent_schema_version": (
                "v3" if self.run_ownership_identity is not None else "v2"
            ),
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
        if self.run_ownership_identity is not None:
            payload["run_ownership_identity"] = identity_fields(
                self.run_ownership_identity
            )
        return payload

    def terminal_payload(
        self,
        status: RunLifecycleStatus,
        *,
        failure_code: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "intent_schema_version": (
                "v3" if self.run_ownership_identity is not None else "v2"
            ),
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
        if self.run_ownership_identity is not None:
            payload["run_ownership_identity"] = identity_fields(
                self.run_ownership_identity
            )
        return payload


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

    def reserve_replay_attempt(self, candidate_key: str) -> int:
        """Durably reserve one candidate ordinal under a cross-process lease."""

        try:
            require_identifier("candidate_key", candidate_key)
        except RecordValidationError:
            raise ApplicationServiceError("replay_candidate_invalid") from None
        deadline = time.monotonic() + _REPLAY_PLANNING_WAIT_SECONDS
        while True:
            try:
                with self._claim_replay_execution(f"planning-{candidate_key}"):
                    return self._reserve_replay_attempt_locked(candidate_key)
            except ApplicationServiceError as exc:
                if exc.code != "replay_execution_busy":
                    raise
                if time.monotonic() >= deadline:
                    raise ApplicationServiceError("replay_planning_busy") from None
                time.sleep(_REPLAY_PLANNING_RETRY_SECONDS)

    def _reserve_replay_attempt_locked(self, candidate_key: str) -> int:
        prefix = f"run-replay-{candidate_key}-"
        planning_stream = f"planning-replay-{candidate_key}"
        try:
            records = tuple(stored.record for stored in self.store.iter_records())
            streams = self.store.iter_streams()
        except EvidenceError as exc:
            raise ApplicationServiceError(exc.code) from None
        used = {
            record.run_id
            for record in records
            if isinstance(record, Run) and record.run_id.startswith(prefix)
        }
        for snapshot in streams:
            for event in snapshot.events:
                if event.event_type == "replay_execution_started":
                    if event.payload.get("candidate_key") != candidate_key:
                        continue
                    run_id = event.payload.get("run_id")
                    if not isinstance(run_id, str) or not run_id.startswith(prefix):
                        raise ApplicationServiceError(
                            "replay_planning_evidence_conflict"
                        )
                    used.add(run_id)
                    continue
                if event.event_type != "replay_planning_reserved":
                    continue
                if snapshot.stream_id != planning_stream:
                    if event.payload.get("candidate_key") == candidate_key:
                        raise ApplicationServiceError(
                            "replay_planning_evidence_conflict"
                        )
                    continue
                payload = event.payload
                ordinal = payload.get("ordinal")
                run_id = payload.get("run_id")
                if (
                    set(payload)
                    != {"schema_version", "candidate_key", "ordinal", "run_id"}
                    or payload.get("schema_version") != "v1"
                    or payload.get("candidate_key") != candidate_key
                    or type(ordinal) is not int
                    or ordinal < 1
                    or run_id != f"{prefix}{ordinal:03d}"
                    or event.idempotency_key
                    != f"{planning_stream}-reserved-{ordinal:03d}"
                ):
                    raise ApplicationServiceError("replay_planning_evidence_conflict")
                used.add(run_id)
        ordinal = 1
        while f"{prefix}{ordinal:03d}" in used:
            ordinal += 1
        run_id = f"{prefix}{ordinal:03d}"
        request = EventRequest(
            f"{planning_stream}-reserved-{ordinal:03d}",
            "replay_planning_reserved",
            {
                "schema_version": "v1",
                "candidate_key": candidate_key,
                "ordinal": ordinal,
                "run_id": run_id,
            },
        )
        self._append_replay_event(planning_stream, request)
        durable = tuple(
            event
            for event in self._replay_events(planning_stream)
            if event.idempotency_key == request.idempotency_key
        )
        if (
            len(durable) != 1
            or durable[0].request_fields() != request.canonical_fields()
        ):
            raise ApplicationServiceError("replay_planning_evidence_conflict")
        return ordinal

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
            run_ownership_identity=(
                run_service.planned_first_attempt_ownership(launch)
            ),
            artifact_id=launch.artifact_id,
            source_experiment_id=plan.source_experiment.experiment_id,
            source_experiment_identity=plan.source_experiment.identity,
            planned_experiment_id=plan.planned_experiment.experiment_id,
            planned_experiment_identity=plan.planned_experiment.identity,
            source_manifest_identity=plan.source_experiment.manifest_identity,
            planned_manifest_identity=plan.planned_experiment.manifest_identity,
        )
        with self._claim_replay_execution(intent.stream_id):
            return self._execute_locked(
                request,
                intent,
                run_service,
                expected_request,
                expected_run,
                control=control,
            )

    def _execute_locked(
        self,
        request: ReplayExecutionRequest,
        intent: _ReplayIntent,
        run_service: RunService,
        expected_request: ResolvedRuntimeRequest,
        expected_run: Run,
        *,
        control: FixtureControl | None,
    ) -> ReplayExecutionResult | ReplayReconciliationResult:
        """Execute or reconcile while holding the exact replay stream lease."""

        plan = request.plan
        launch = request.launch
        events = self._replay_events(intent.stream_id)
        durable_intent, terminal = self._validate_replay_events(
            events, expected_intent=intent
        )
        if durable_intent is not None:
            intent = durable_intent
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
            reconciled = self._reconcile_intent(intent, run_service, abandoned=True)
            if reconciled is not None:
                self._append_terminal(
                    intent,
                    reconciled.status,
                    failure_code=reconciled.failure_code,
                )
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
            if exc.code == "run_ownership_unavailable" and (
                self._run_status_or_none(run_service, intent.run_id)
                is RunLifecycleStatus.RUNNING
            ):
                raise ApplicationServiceError("replay_execution_active") from None
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
        run_service.reconcile_terminal_ownership(
            intent.run_id,
            intent.run_ownership_identity,
            artifact_id=intent.artifact_id,
        )
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
            try:
                with self._claim_replay_execution(snapshot.stream_id):
                    current = self._replay_events(snapshot.stream_id)
                    intent, terminal = self._validate_replay_events(current)
                    if intent is None:
                        continue
                    if snapshot.stream_id != intent.stream_id:
                        raise ApplicationServiceError(
                            "replay_execution_evidence_conflict"
                        )
                    if terminal is not None:
                        results.append(terminal)
                        continue
                    reconciled = self._reconcile_intent(
                        intent, run_service, abandoned=True
                    )
                    if reconciled is None:
                        continue
                    self._append_terminal(
                        intent,
                        reconciled.status,
                        failure_code=reconciled.failure_code,
                    )
                    results.append(reconciled)
            except ApplicationServiceError as exc:
                if exc.code == "replay_execution_busy":
                    continue
                raise
        return tuple(results)

    def reconcile_plan(
        self,
        plan_id: str,
        run_id: str,
        *,
        candidate_key: str | None = None,
        mode: str | None = None,
    ) -> ReplayReconciliationResult:
        """Return one exact durable terminal after process-local draft loss."""

        try:
            require_identifier("plan_id", plan_id)
        except RecordValidationError:
            raise ApplicationServiceError("replay_plan_mismatch") from None
        try:
            require_identifier("run_id", run_id)
        except RecordValidationError:
            raise ApplicationServiceError("replay_run_plan_mismatch") from None
        results = self.reconcile_pending()
        matches = tuple(
            item
            for item in results
            if item.plan_id == plan_id and item.run_id == run_id
        )
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
        *,
        abandoned: bool = False,
    ) -> ReplayReconciliationResult | None:
        run_service.reconcile_launch_record_set(
            intent.run_id,
            intent.runtime_request_identity,
            intent.run_identity,
        )
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
        if len(source_experiments) != 1 or len(planned_experiments) != 1:
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        if not runs and not requests:
            if not abandoned:
                return None
            run_service.reconcile_unlaunched_ownership(
                intent.run_id,
                intent.runtime_request_identity,
                intent.run_ownership_identity,
            )
            return ReplayReconciliationResult(
                intent.plan_id,
                intent.mode,
                intent.candidate_key,
                intent.run_id,
                intent.run_identity,
                intent.runtime_request_identity,
                RunLifecycleStatus.FAILED,
                "run_launch_record_persistence_failed",
            )
        if len(runs) != 1 or len(requests) != 1:
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
        if status is RunLifecycleStatus.RUNNING and abandoned:
            status = run_service.reconcile_abandoned_running(
                intent.run_id,
                intent.run_ownership_identity,
                artifact_id=intent.artifact_id,
            )
        if status is None or status is RunLifecycleStatus.RUNNING:
            return None
        events = run_service._events(intent.run_id)
        launched = tuple(
            event for event in events if event.event_type == "run_launched"
        )
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
        launch_evidence_failed = (
            status is RunLifecycleStatus.FAILED
            and not launched
            and terminal.payload.get("phase") in {"launch_evidence", "launch_records"}
        )
        if not launch_evidence_failed and (
            len(launched) != 1
            or _payload_identity(launched[0].payload, "run_identity")
            != intent.run_identity
            or _payload_identity(launched[0].payload, "runtime_request_identity")
            != intent.runtime_request_identity
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")
        if launch_evidence_failed and len(launched) != 0:
            raise ApplicationServiceError("replay_execution_evidence_conflict")
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
        run_service.reconcile_terminal_ownership(
            intent.run_id,
            intent.run_ownership_identity,
            artifact_id=intent.artifact_id,
        )
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
            if exc.code in {"run_not_found", "run_lifecycle_incomplete"}:
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
        if expected_intent is not None and not _intents_compatible(
            intent, expected_intent
        ):
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
            try:
                matches = tuple(
                    event
                    for event in self._replay_events(stream_id)
                    if event.idempotency_key == request.idempotency_key
                )
            except ApplicationServiceError:
                matches = ()
            if len(matches) == 1 and matches[0].request_fields() == (
                request.canonical_fields()
            ):
                return
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
        _, existing = self._validate_replay_events(
            self._replay_events(intent.stream_id), expected_intent=intent
        )
        if existing is not None:
            if existing.status == status and existing.failure_code == failure_code:
                return
            raise ApplicationServiceError("replay_execution_evidence_conflict")
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
        _, durable = self._validate_replay_events(
            self._replay_events(intent.stream_id), expected_intent=intent
        )
        if (
            durable is None
            or durable.status != status
            or durable.failure_code != failure_code
        ):
            raise ApplicationServiceError("replay_execution_evidence_conflict")

    @contextmanager
    def _claim_replay_execution(self, stream_id: str) -> Iterator[None]:
        """Hold one non-blocking cross-process lease for an exact replay stream."""

        try:
            require_identifier("replay_stream_id", stream_id)
        except RecordValidationError:
            raise ApplicationServiceError(
                "replay_execution_evidence_conflict"
            ) from None
        lock_path = (
            Path(os.path.abspath(self.project_root / RUNTIME_OUTPUT_DIRECTORY))
            / "runtime-ownership"
            / ".control"
            / "replay"
            / stream_id
            / "lease.lock"
        )
        try:
            ensure_safe_directory(lock_path.parent)
            existing = safe_path_stat(lock_path, allow_missing=True)
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ApplicationServiceError("replay_execution_lock_invalid")
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(lock_path, flags, 0o600)
        except ApplicationServiceError:
            raise
        except (OSError, UnsafeFilesystemPath):
            raise ApplicationServiceError("replay_execution_lock_unavailable") from None
        with os.fdopen(descriptor, "r+b") as handle:
            locked = False
            try:
                opened = os.fstat(handle.fileno())
                current = require_safe_regular_file(lock_path)
                if (
                    is_link_or_reparse(opened)
                    or not stat.S_ISREG(opened.st_mode)
                    or not same_file_object(opened, current)
                    or opened.st_nlink != 1
                ):
                    raise ApplicationServiceError("replay_execution_lock_invalid")
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                elif size != 1:
                    raise ApplicationServiceError("replay_execution_lock_invalid")
                handle.seek(0)
                _lock_replay_handle(handle)
                locked = True
                opened = os.fstat(handle.fileno())
                current = require_safe_regular_file(lock_path)
                handle.seek(0)
                if (
                    not same_file_object(opened, current)
                    or opened.st_nlink != 1
                    or opened.st_size != 1
                    or handle.read(1) != b"\0"
                ):
                    raise ApplicationServiceError("replay_execution_lock_invalid")
                yield
            except ApplicationServiceError:
                raise
            except (OSError, UnsafeFilesystemPath):
                raise ApplicationServiceError(
                    "replay_execution_lock_unavailable"
                ) from None
            finally:
                if locked:
                    try:
                        handle.seek(0)
                        _unlock_replay_handle(handle)
                    except OSError:
                        pass


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
    schema_version = payload.get("intent_schema_version")
    if schema_version == "v3":
        required.add("run_ownership_identity")
    elif schema_version != "v2":
        raise ApplicationServiceError("replay_execution_evidence_conflict")
    if set(payload) != required:
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
            run_ownership_identity=(
                _payload_identity(payload, "run_ownership_identity")
                if schema_version == "v3"
                else None
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


def _intents_compatible(first: _ReplayIntent, second: _ReplayIntent) -> bool:
    """Permit a legacy v2 intent only when every shared identity is exact."""

    if first == second:
        return True
    first_without_claim = dataclass_replace(first, run_ownership_identity=None)
    second_without_claim = dataclass_replace(second, run_ownership_identity=None)
    return first_without_claim == second_without_claim and (
        first.run_ownership_identity is None
        or second.run_ownership_identity is None
        or first.run_ownership_identity == second.run_ownership_identity
    )


def _lock_replay_handle(handle: Any) -> None:
    try:
        if os.name == "nt":
            msvcrt: Any = importlib.import_module("msvcrt")
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        deadlock = getattr(errno, "EDEADLK", None)
        if (
            isinstance(exc, BlockingIOError)
            or exc.errno in {errno.EACCES, errno.EAGAIN}
            or (deadlock is not None and exc.errno == deadlock)
        ):
            raise ApplicationServiceError("replay_execution_busy") from None
        raise ApplicationServiceError("replay_execution_lock_unavailable") from None


def _unlock_replay_handle(handle: Any) -> None:
    if os.name == "nt":
        msvcrt: Any = importlib.import_module("msvcrt")
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl: Any = importlib.import_module("fcntl")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
