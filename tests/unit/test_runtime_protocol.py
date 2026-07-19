from concurrent.futures import ThreadPoolExecutor

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.controller import (
    ControllerState,
    RuntimeControllerError,
    SerializedResourceCoordinator,
    SerializedRunController,
)
from temper_ml.runtime.protocol import (
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeProtocolError,
)


IDENTITY = ContentIdentity("sha256", "1" * 64)
BUNDLE = ContentIdentity("sha256", "2" * 64)


def _message(
    sequence: int,
    kind: RuntimeMessageKind,
    payload: dict[str, object],
) -> RuntimeMessage:
    return RuntimeMessage(IDENTITY, "run-protocol", sequence, kind, payload)


def test_messages_round_trip_and_reject_private_or_malformed_payloads() -> None:
    launched = _message(
        1,
        RuntimeMessageKind.LAUNCHED,
        {"operation": "train", "target_class": "wsl_rocm"},
    )
    assert RuntimeMessage.from_bytes(launched.to_bytes()) == launched

    with pytest.raises(RuntimeProtocolError, match="runtime_message_private_payload"):
        _message(
            1,
            RuntimeMessageKind.LAUNCHED,
            {"operation": "train", "target_class": "C:\\private"},
        )
    with pytest.raises(RuntimeProtocolError, match="runtime_message_private_payload"):
        _message(
            1,
            RuntimeMessageKind.FAILED,
            {
                "terminal": True,
                "failure_code": "worker_failed",
                "phase": "runtime",
                "hostname": "private-host",
            },
        )
    with pytest.raises(
        RuntimeProtocolError, match="runtime_message_payload_fields_invalid"
    ):
        _message(1, RuntimeMessageKind.PROGRESS, {"step": 1})


def test_controller_replays_exact_messages_and_reconstructs_reconnection() -> None:
    messages = (
        _message(
            1,
            RuntimeMessageKind.LAUNCHED,
            {"operation": "train", "target_class": "wsl_rocm"},
        ),
        _message(
            2,
            RuntimeMessageKind.PROGRESS,
            {"step": 1, "total_steps": 2},
        ),
        _message(
            3,
            RuntimeMessageKind.METRIC,
            {"name": "training_loss", "value_microunits": 100, "step": 1},
        ),
        _message(
            4,
            RuntimeMessageKind.HEARTBEAT,
            {"step": 1, "state": "training"},
        ),
        _message(
            5,
            RuntimeMessageKind.DISCONNECTED,
            {"last_received_sequence": 4},
        ),
        _message(
            6,
            RuntimeMessageKind.RECONNECTED,
            {"resume_after_sequence": 5},
        ),
        _message(
            7,
            RuntimeMessageKind.ARTIFACT_READY,
            {
                "bundle_identity": {
                    "algorithm": BUNDLE.algorithm,
                    "value": BUNDLE.value,
                },
                "member_count": 3,
            },
        ),
        _message(
            8,
            RuntimeMessageKind.COMPLETED,
            {"terminal": True, "verified_transfer": True},
        ),
    )
    controller = SerializedRunController(IDENTITY, "run-protocol")
    for message in messages:
        controller.accept(message)
    assert controller.accept(messages[-1]) == controller.snapshot()
    snapshot = controller.snapshot()
    assert snapshot.state is ControllerState.COMPLETED
    assert snapshot.last_sequence == 8
    assert snapshot.last_step == 1
    assert snapshot.metric_count == 1
    assert snapshot.heartbeat_count == 1
    assert snapshot.artifact_identity == BUNDLE
    assert (
        SerializedRunController.reconstruct(
            IDENTITY, "run-protocol", messages
        ).snapshot()
        == snapshot
    )

    conflicting = _message(
        8,
        RuntimeMessageKind.FAILED,
        {"terminal": True, "failure_code": "conflict", "phase": "runtime"},
    )
    with pytest.raises(
        RuntimeControllerError, match="controller_message_replay_conflict"
    ):
        controller.accept(conflicting)


def test_controller_serializes_terminal_races_and_resource_ownership() -> None:
    controller = SerializedRunController(IDENTITY, "run-protocol")
    controller.accept(
        _message(
            1,
            RuntimeMessageKind.LAUNCHED,
            {"operation": "train", "target_class": "native_rocm"},
        )
    )
    controller.accept(
        _message(
            2,
            RuntimeMessageKind.CANCELLATION_REQUESTED,
            {"acknowledged": True},
        )
    )
    controller.accept(_message(3, RuntimeMessageKind.CANCELLED, {"terminal": True}))
    with pytest.raises(
        RuntimeControllerError, match="controller_message_after_terminal"
    ):
        controller.accept(
            _message(
                4,
                RuntimeMessageKind.FAILED,
                {
                    "terminal": True,
                    "failure_code": "late_failure",
                    "phase": "runtime",
                },
            )
        )

    resources = SerializedResourceCoordinator({"accelerator-0": 1})

    def acquire(owner: str) -> str:
        try:
            resources.acquire(owner, ("accelerator-0",))
            return "acquired"
        except RuntimeControllerError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(acquire, ("run-one", "run-two")))
    assert sorted(outcomes) == ["acquired", "resource_unavailable"]
    owner = next(iter(resources.leases))
    resources.release(owner)
    assert not resources.leases
