"""Typed, public-safe messages for the local Temper worker boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Mapping

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.records import (
    RecordValidationError,
    freeze_json_object,
    identity_fields,
    parse_identity,
    require_identifier,
    thaw_json,
)
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json

RUNTIME_PROTOCOL_VERSION = "v1"
RUNTIME_MESSAGE_PROJECTION = HashProjection("runtime.worker_message", "v1")

_PRIVATE_KEY = re.compile(
    r"(?:^|_)(?:absolute_path|host(?:name)?|user(?:name)?|machine_id|device_id|"
    r"process_id|pid|ip|mac|home|account|private_url)(?:_|$)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class RuntimeProtocolError(RuntimeError):
    """A stable worker-protocol validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RuntimeOperation(str, Enum):
    """Operations supported by the same local runtime port."""

    PROBE = "probe"
    TRAIN = "train"
    EVALUATE = "evaluate"
    INFER_FOCUSED = "infer_focused"
    INFER_BATCH = "infer_batch"


class RuntimeMessageKind(str, Enum):
    """Lifecycle and evidence messages admitted across the worker boundary."""

    LAUNCHED = "launched"
    PROGRESS = "progress"
    METRIC = "metric"
    CHECKPOINT = "checkpoint"
    LOG = "log"
    HEARTBEAT = "heartbeat"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    DISCONNECTED = "disconnected"
    RECONNECTED = "reconnected"
    ARTIFACT_READY = "artifact_ready"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            RuntimeMessageKind.CANCELLED,
            RuntimeMessageKind.INTERRUPTED,
            RuntimeMessageKind.COMPLETED,
            RuntimeMessageKind.FAILED,
        }


@dataclass(frozen=True)
class RuntimeMessage:
    """One ordered message with a content identity independent of transport."""

    request_identity: ContentIdentity
    run_id: str
    sequence: int
    kind: RuntimeMessageKind
    payload: Mapping[str, Any]
    protocol_version: str = RUNTIME_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != RUNTIME_PROTOCOL_VERSION:
            raise RuntimeProtocolError("runtime_protocol_version_unsupported")
        if not isinstance(self.request_identity, ContentIdentity):
            raise RuntimeProtocolError("runtime_message_request_identity_invalid")
        try:
            require_identifier("run_id", self.run_id)
        except RecordValidationError:
            raise RuntimeProtocolError("runtime_message_run_id_invalid") from None
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise RuntimeProtocolError("runtime_message_sequence_invalid")
        if not isinstance(self.kind, RuntimeMessageKind):
            raise RuntimeProtocolError("runtime_message_kind_invalid")
        try:
            frozen = freeze_json_object(self.payload, field="runtime_message.payload")
        except (RecordValidationError, TypeError, ValueError):
            raise RuntimeProtocolError("runtime_message_payload_invalid") from None
        value = thaw_json(frozen)
        _reject_private_value(value, "payload")
        _validate_payload(self.kind, value)
        object.__setattr__(self, "payload", frozen)

    @property
    def identity(self) -> ContentIdentity:
        return content_identity(RUNTIME_MESSAGE_PROJECTION, self.projected_fields())

    def projected_fields(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_identity": identity_fields(self.request_identity),
            "run_id": self.run_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "payload": thaw_json(self.payload),
        }

    def to_dict(self) -> dict[str, object]:
        value = self.projected_fields()
        value["identity"] = identity_fields(self.identity)
        return value

    def to_bytes(self) -> bytes:
        return dumps_canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeMessage":
        expected = {
            "protocol_version",
            "request_identity",
            "run_id",
            "sequence",
            "kind",
            "payload",
            "identity",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise RuntimeProtocolError("runtime_message_fields_invalid")
        try:
            request_identity = parse_identity(
                value["request_identity"],
                field="request_identity",  # type: ignore[arg-type]
            )
            claimed = parse_identity(value["identity"], field="identity")  # type: ignore[arg-type]
            message = cls(
                request_identity=request_identity,
                run_id=value["run_id"],  # type: ignore[arg-type]
                sequence=value["sequence"],  # type: ignore[arg-type]
                kind=RuntimeMessageKind(value["kind"]),
                payload=value["payload"],  # type: ignore[arg-type]
                protocol_version=value["protocol_version"],  # type: ignore[arg-type]
            )
        except RuntimeProtocolError:
            raise
        except (KeyError, RecordValidationError, TypeError, ValueError):
            raise RuntimeProtocolError("runtime_message_fields_invalid") from None
        if message.identity != claimed:
            raise RuntimeProtocolError("runtime_message_identity_mismatch")
        return message

    @classmethod
    def from_bytes(cls, data: bytes) -> "RuntimeMessage":
        try:
            value = loads_canonical_json(data)
        except (TypeError, ValueError):
            raise RuntimeProtocolError("runtime_message_json_invalid") from None
        if not isinstance(value, Mapping):
            raise RuntimeProtocolError("runtime_message_json_invalid")
        return cls.from_dict(value)


@dataclass(frozen=True)
class RuntimeMessageLedger:
    """An immutable ordered message set suitable for reconnect reconciliation."""

    request_identity: ContentIdentity
    run_id: str
    messages: tuple[RuntimeMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_identity, ContentIdentity):
            raise RuntimeProtocolError("runtime_ledger_request_identity_invalid")
        try:
            require_identifier("run_id", self.run_id)
        except RecordValidationError:
            raise RuntimeProtocolError("runtime_ledger_run_id_invalid") from None
        if not isinstance(self.messages, tuple):
            raise RuntimeProtocolError("runtime_ledger_messages_invalid")
        for expected_sequence, message in enumerate(self.messages, 1):
            if (
                not isinstance(message, RuntimeMessage)
                or message.request_identity != self.request_identity
                or message.run_id != self.run_id
                or message.sequence != expected_sequence
            ):
                raise RuntimeProtocolError("runtime_ledger_messages_invalid")

    @property
    def by_sequence(self) -> Mapping[int, RuntimeMessage]:
        return MappingProxyType(
            {message.sequence: message for message in self.messages}
        )


def _reject_private_value(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _PRIVATE_KEY.search(key):
                raise RuntimeProtocolError("runtime_message_private_payload")
            _reject_private_value(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_value(item, f"{path}[{index}]")
        return
    if isinstance(value, str) and (
        value.startswith(("/", "\\\\"))
        or _WINDOWS_ABSOLUTE.match(value)
        or "://" in value
    ):
        raise RuntimeProtocolError("runtime_message_private_payload")


def _validate_payload(kind: RuntimeMessageKind, value: dict[str, Any]) -> None:
    required: dict[RuntimeMessageKind, set[str]] = {
        RuntimeMessageKind.LAUNCHED: {"operation", "target_class"},
        RuntimeMessageKind.PROGRESS: {"step", "total_steps"},
        RuntimeMessageKind.METRIC: {"name", "value_microunits", "step"},
        RuntimeMessageKind.CHECKPOINT: {
            "step",
            "checkpoint_identity",
            "training_state_identity",
            "byte_count",
            "resume_compatible",
        },
        RuntimeMessageKind.LOG: {"code", "ordinal", "step"},
        RuntimeMessageKind.HEARTBEAT: {"step", "state"},
        RuntimeMessageKind.CANCELLATION_REQUESTED: {"acknowledged"},
        RuntimeMessageKind.CANCELLED: {"terminal"},
        RuntimeMessageKind.INTERRUPTED: {
            "terminal",
            "recovery_checkpoint_count",
        },
        RuntimeMessageKind.DISCONNECTED: {"last_received_sequence"},
        RuntimeMessageKind.RECONNECTED: {"resume_after_sequence"},
        RuntimeMessageKind.ARTIFACT_READY: {"bundle_identity", "member_count"},
        RuntimeMessageKind.COMPLETED: {"terminal", "verified_transfer"},
        RuntimeMessageKind.FAILED: {"terminal", "failure_code", "phase"},
    }
    if set(value) != required[kind]:
        raise RuntimeProtocolError("runtime_message_payload_fields_invalid")
    if kind is RuntimeMessageKind.LAUNCHED:
        try:
            RuntimeOperation(value["operation"])
            require_identifier("target_class", value["target_class"])
        except (RecordValidationError, TypeError, ValueError):
            raise RuntimeProtocolError("runtime_message_payload_invalid") from None
    elif kind in {RuntimeMessageKind.PROGRESS, RuntimeMessageKind.HEARTBEAT}:
        _non_negative_int(value["step"])
        if kind is RuntimeMessageKind.PROGRESS:
            total = _positive_int(value["total_steps"])
            if value["step"] > total:
                raise RuntimeProtocolError("runtime_message_payload_invalid")
        elif not isinstance(value["state"], str) or not value["state"]:
            raise RuntimeProtocolError("runtime_message_payload_invalid")
    elif kind is RuntimeMessageKind.METRIC:
        try:
            require_identifier("name", value["name"])
        except RecordValidationError:
            raise RuntimeProtocolError("runtime_message_payload_invalid") from None
        _non_negative_int(value["step"])
        if isinstance(value["value_microunits"], bool) or not isinstance(
            value["value_microunits"], int
        ):
            raise RuntimeProtocolError("runtime_message_payload_invalid")
    elif kind is RuntimeMessageKind.CHECKPOINT:
        _positive_int(value["step"])
        _positive_int(value["byte_count"])
        _identity_mapping(value["checkpoint_identity"])
        _identity_mapping(value["training_state_identity"])
        if not isinstance(value["resume_compatible"], bool):
            raise RuntimeProtocolError("runtime_message_payload_invalid")
    elif kind is RuntimeMessageKind.LOG:
        _positive_int(value["ordinal"])
        _non_negative_int(value["step"])
        try:
            require_identifier("code", value["code"])
        except RecordValidationError:
            raise RuntimeProtocolError("runtime_message_payload_invalid") from None
    elif kind in {
        RuntimeMessageKind.CANCELLATION_REQUESTED,
        RuntimeMessageKind.CANCELLED,
    }:
        expected = (
            "acknowledged"
            if kind is RuntimeMessageKind.CANCELLATION_REQUESTED
            else "terminal"
        )
        if value[expected] is not True:
            raise RuntimeProtocolError("runtime_message_payload_invalid")
    elif kind is RuntimeMessageKind.INTERRUPTED:
        if value["terminal"] is not True:
            raise RuntimeProtocolError("runtime_message_payload_invalid")
        _non_negative_int(value["recovery_checkpoint_count"])
    elif kind in {RuntimeMessageKind.DISCONNECTED, RuntimeMessageKind.RECONNECTED}:
        key = (
            "last_received_sequence"
            if kind is RuntimeMessageKind.DISCONNECTED
            else "resume_after_sequence"
        )
        _non_negative_int(value[key])
    elif kind is RuntimeMessageKind.ARTIFACT_READY:
        _identity_mapping(value["bundle_identity"])
        _positive_int(value["member_count"])
    elif kind is RuntimeMessageKind.COMPLETED:
        if value["terminal"] is not True or value["verified_transfer"] is not True:
            raise RuntimeProtocolError("runtime_message_payload_invalid")
    elif kind is RuntimeMessageKind.FAILED:
        if value["terminal"] is not True:
            raise RuntimeProtocolError("runtime_message_payload_invalid")
        for field in ("failure_code", "phase"):
            try:
                require_identifier(field, value[field])
            except RecordValidationError:
                raise RuntimeProtocolError("runtime_message_payload_invalid") from None


def _identity_mapping(value: Any) -> ContentIdentity:
    if not isinstance(value, Mapping):
        raise RuntimeProtocolError("runtime_message_payload_invalid")
    try:
        return parse_identity(value, field="runtime_message.identity")
    except RecordValidationError:
        raise RuntimeProtocolError("runtime_message_payload_invalid") from None


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeProtocolError("runtime_message_payload_invalid")
    return value


def _positive_int(value: Any) -> int:
    result = _non_negative_int(value)
    if result == 0:
        raise RuntimeProtocolError("runtime_message_payload_invalid")
    return result
