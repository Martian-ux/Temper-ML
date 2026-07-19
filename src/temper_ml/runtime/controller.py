"""Serialized, rebuildable ownership for live local runtime entities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Mapping

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import RecordValidationError, require_identifier
from temper_ml.runtime.protocol import (
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeProtocolError,
)


class RuntimeControllerError(RuntimeError):
    """A stable serialized-ownership failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ControllerState(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    CANCELLING = "cancelling"
    DISCONNECTED = "disconnected"
    ARTIFACT_READY = "artifact_ready"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            ControllerState.CANCELLED,
            ControllerState.INTERRUPTED,
            ControllerState.COMPLETED,
            ControllerState.FAILED,
        }


@dataclass(frozen=True)
class ControllerSnapshot:
    request_identity: ContentIdentity
    run_id: str
    state: ControllerState
    last_sequence: int
    last_step: int
    heartbeat_count: int
    checkpoint_count: int
    metric_count: int
    artifact_identity: ContentIdentity | None
    terminal_message_identity: ContentIdentity | None


class SerializedRunController:
    """One live non-canonical owner for one immutable runtime request."""

    def __init__(self, request_identity: ContentIdentity, run_id: str) -> None:
        if not isinstance(request_identity, ContentIdentity):
            raise RuntimeControllerError("controller_request_identity_invalid")
        try:
            require_identifier("run_id", run_id)
        except RecordValidationError:
            raise RuntimeControllerError("controller_run_id_invalid") from None
        self._request_identity = request_identity
        self._run_id = run_id
        self._state = ControllerState.CREATED
        self._last_sequence = 0
        self._last_step = 0
        self._heartbeat_count = 0
        self._checkpoint_count = 0
        self._metric_count = 0
        self._artifact_identity: ContentIdentity | None = None
        self._terminal_identity: ContentIdentity | None = None
        self._messages: dict[int, RuntimeMessage] = {}
        self._state_before_disconnect: ControllerState | None = None
        self._lock = RLock()

    @classmethod
    def reconstruct(
        cls,
        request_identity: ContentIdentity,
        run_id: str,
        messages: Iterable[RuntimeMessage],
    ) -> "SerializedRunController":
        controller = cls(request_identity, run_id)
        for message in messages:
            controller.accept(message)
        return controller

    def accept(self, message: RuntimeMessage) -> ControllerSnapshot:
        """Apply one exact next message or idempotently accept an exact replay."""

        if not isinstance(message, RuntimeMessage):
            raise RuntimeControllerError("controller_message_invalid")
        with self._lock:
            if (
                message.request_identity != self._request_identity
                or message.run_id != self._run_id
            ):
                raise RuntimeControllerError("controller_message_subject_mismatch")
            prior = self._messages.get(message.sequence)
            if prior is not None:
                if prior.identity != message.identity:
                    raise RuntimeControllerError("controller_message_replay_conflict")
                return self.snapshot()
            if message.sequence != self._last_sequence + 1:
                raise RuntimeControllerError("controller_message_sequence_gap")
            if self._state.terminal:
                raise RuntimeControllerError("controller_message_after_terminal")
            self._transition(message)
            self._messages[message.sequence] = message
            self._last_sequence = message.sequence
            if self._state.terminal:
                self._terminal_identity = message.identity
            return self.snapshot()

    def snapshot(self) -> ControllerSnapshot:
        with self._lock:
            return ControllerSnapshot(
                request_identity=self._request_identity,
                run_id=self._run_id,
                state=self._state,
                last_sequence=self._last_sequence,
                last_step=self._last_step,
                heartbeat_count=self._heartbeat_count,
                checkpoint_count=self._checkpoint_count,
                metric_count=self._metric_count,
                artifact_identity=self._artifact_identity,
                terminal_message_identity=self._terminal_identity,
            )

    @property
    def messages(self) -> tuple[RuntimeMessage, ...]:
        with self._lock:
            return tuple(self._messages[index] for index in sorted(self._messages))

    def _transition(self, message: RuntimeMessage) -> None:
        kind = message.kind
        payload = message.payload
        if kind is RuntimeMessageKind.LAUNCHED:
            if self._state is not ControllerState.CREATED:
                raise RuntimeControllerError("controller_duplicate_launch")
            self._state = ControllerState.ACTIVE
            return
        if self._state is ControllerState.CREATED:
            raise RuntimeControllerError("controller_launch_missing")
        if kind in {
            RuntimeMessageKind.PROGRESS,
            RuntimeMessageKind.METRIC,
            RuntimeMessageKind.CHECKPOINT,
            RuntimeMessageKind.LOG,
            RuntimeMessageKind.HEARTBEAT,
        }:
            if self._state not in {
                ControllerState.ACTIVE,
                ControllerState.CANCELLING,
            }:
                raise RuntimeControllerError("controller_activity_in_invalid_state")
            if kind is RuntimeMessageKind.PROGRESS:
                step = int(payload["step"])
                if step < self._last_step:
                    raise RuntimeControllerError("controller_progress_regressed")
                self._last_step = step
            elif kind is RuntimeMessageKind.METRIC:
                self._metric_count += 1
            elif kind is RuntimeMessageKind.CHECKPOINT:
                step = int(payload["step"])
                if step < self._last_step:
                    raise RuntimeControllerError("controller_checkpoint_regressed")
                self._last_step = step
                self._checkpoint_count += 1
            elif kind is RuntimeMessageKind.HEARTBEAT:
                self._heartbeat_count += 1
            return
        if kind is RuntimeMessageKind.CANCELLATION_REQUESTED:
            if self._state is not ControllerState.ACTIVE:
                raise RuntimeControllerError("controller_cancellation_invalid")
            self._state = ControllerState.CANCELLING
            return
        if kind is RuntimeMessageKind.DISCONNECTED:
            if self._state not in {
                ControllerState.ACTIVE,
                ControllerState.CANCELLING,
            }:
                raise RuntimeControllerError("controller_disconnect_invalid")
            if int(payload["last_received_sequence"]) != self._last_sequence:
                raise RuntimeControllerError("controller_disconnect_sequence_mismatch")
            self._state_before_disconnect = self._state
            self._state = ControllerState.DISCONNECTED
            return
        if kind is RuntimeMessageKind.RECONNECTED:
            if self._state is not ControllerState.DISCONNECTED:
                raise RuntimeControllerError("controller_reconnect_invalid")
            if int(payload["resume_after_sequence"]) != self._last_sequence:
                raise RuntimeControllerError("controller_reconnect_sequence_mismatch")
            if self._state_before_disconnect not in {
                ControllerState.ACTIVE,
                ControllerState.CANCELLING,
            }:
                raise RuntimeControllerError("controller_reconnect_state_missing")
            self._state = self._state_before_disconnect
            self._state_before_disconnect = None
            return
        if kind is RuntimeMessageKind.ARTIFACT_READY:
            if self._state is not ControllerState.ACTIVE:
                raise RuntimeControllerError("controller_artifact_invalid")
            raw = payload["bundle_identity"]
            if not isinstance(raw, Mapping):
                raise RuntimeControllerError("controller_artifact_invalid")
            try:
                from temper_ml.domain.records import parse_identity

                self._artifact_identity = parse_identity(raw, field="bundle_identity")
            except (RecordValidationError, RuntimeProtocolError):
                raise RuntimeControllerError("controller_artifact_invalid") from None
            self._state = ControllerState.ARTIFACT_READY
            return
        if kind is RuntimeMessageKind.COMPLETED:
            if self._state is not ControllerState.ARTIFACT_READY:
                raise RuntimeControllerError("controller_completion_without_artifact")
            self._state = ControllerState.COMPLETED
            return
        if kind is RuntimeMessageKind.CANCELLED:
            if self._state is not ControllerState.CANCELLING:
                raise RuntimeControllerError("controller_cancelled_without_request")
            self._state = ControllerState.CANCELLED
            return
        if kind is RuntimeMessageKind.INTERRUPTED:
            if self._state not in {
                ControllerState.ACTIVE,
                ControllerState.CANCELLING,
                ControllerState.DISCONNECTED,
            }:
                raise RuntimeControllerError("controller_interruption_invalid")
            self._state = ControllerState.INTERRUPTED
            return
        if kind is RuntimeMessageKind.FAILED:
            self._state = ControllerState.FAILED
            return
        raise RuntimeControllerError("controller_message_kind_unhandled")


@dataclass(frozen=True)
class ResourceLease:
    owner_id: str
    resources: tuple[str, ...]


class SerializedResourceCoordinator:
    """One serialized non-canonical owner for local accelerator resources."""

    def __init__(self, capacities: Mapping[str, int]) -> None:
        if not isinstance(capacities, Mapping) or not capacities:
            raise RuntimeControllerError("resource_capacities_invalid")
        normalized: dict[str, int] = {}
        for resource, capacity in capacities.items():
            try:
                require_identifier("resource", resource)
            except RecordValidationError:
                raise RuntimeControllerError("resource_capacities_invalid") from None
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 1
            ):
                raise RuntimeControllerError("resource_capacities_invalid")
            normalized[resource] = capacity
        self._capacities = MappingProxyType(dict(sorted(normalized.items())))
        self._leases: dict[str, ResourceLease] = {}
        self._lock = RLock()

    def acquire(self, owner_id: str, resources: Iterable[str]) -> ResourceLease:
        try:
            require_identifier("owner_id", owner_id)
        except RecordValidationError:
            raise RuntimeControllerError("resource_owner_invalid") from None
        requested = tuple(sorted(resources))
        if not requested or len(set(requested)) != len(requested):
            raise RuntimeControllerError("resource_request_invalid")
        if any(resource not in self._capacities for resource in requested):
            raise RuntimeControllerError("resource_unknown")
        with self._lock:
            existing = self._leases.get(owner_id)
            if existing is not None:
                if existing.resources != requested:
                    raise RuntimeControllerError("resource_owner_conflict")
                return existing
            usage = {resource: 0 for resource in self._capacities}
            for lease in self._leases.values():
                for resource in lease.resources:
                    usage[resource] += 1
            if any(
                usage[resource] >= self._capacities[resource] for resource in requested
            ):
                raise RuntimeControllerError("resource_unavailable")
            lease = ResourceLease(owner_id, requested)
            self._leases[owner_id] = lease
            return lease

    def release(self, owner_id: str) -> ResourceLease:
        with self._lock:
            try:
                return self._leases.pop(owner_id)
            except KeyError:
                raise RuntimeControllerError("resource_lease_missing") from None

    @property
    def leases(self) -> Mapping[str, ResourceLease]:
        with self._lock:
            return MappingProxyType(dict(self._leases))
