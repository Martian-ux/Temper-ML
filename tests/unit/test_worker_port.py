from pathlib import Path, PurePosixPath

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.library_backend import LibraryExecutionContext
from temper_ml.runtime.paths import PortableLocation
from temper_ml.runtime.protocol import (
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperation,
)
from temper_ml.runtime.staging import (
    TransferDirection,
    build_transfer_manifest,
)
from temper_ml.runtime.worker_port import (
    WorkerInvocation,
    WorkerPortError,
    WorkerResponse,
    WslWorkerLauncher,
    WslWorkerLaunchSpec,
)


IDENTITY = ContentIdentity("sha256", "3" * 64)


def _invocation() -> WorkerInvocation:
    return WorkerInvocation(
        LibraryExecutionContext(
            IDENTITY,
            "probe-worker-port",
            RuntimeOperation.PROBE,
            "wsl_rocm",
        ),
        PurePosixPath("/temper-staging"),
        PortableLocation("probes/session"),
    )


def _terminal_evidence(root: Path, invocation: WorkerInvocation) -> WorkerResponse:
    output = PortableLocation("probes/session/outputs/capability.json")
    manifest = build_transfer_manifest(
        TransferDirection.WORKER_TO_HOST,
        {output: ("capability_profile", b"{}")},
    )
    messages = (
        RuntimeMessage(
            IDENTITY,
            invocation.context.run_id,
            1,
            RuntimeMessageKind.LAUNCHED,
            {"operation": "probe", "target_class": "wsl_rocm"},
        ),
        RuntimeMessage(
            IDENTITY,
            invocation.context.run_id,
            2,
            RuntimeMessageKind.ARTIFACT_READY,
            {
                "bundle_identity": {
                    "algorithm": manifest.identity.algorithm,
                    "value": manifest.identity.value,
                },
                "member_count": 1,
            },
        ),
        RuntimeMessage(
            IDENTITY,
            invocation.context.run_id,
            3,
            RuntimeMessageKind.COMPLETED,
            {"terminal": True, "verified_transfer": True},
        ),
    )
    message_root = root / "messages"
    message_root.mkdir(parents=True, exist_ok=True)
    for message in messages:
        (message_root / f"{message.sequence:08d}.json").write_bytes(message.to_bytes())
    response = WorkerResponse(
        invocation.context,
        "completed",
        manifest,
        {"capability_location": output.to_dict()},
    )
    (root / "response.json").write_bytes(response.to_bytes())
    return response


def test_wsl_command_is_an_exact_no_shell_argv() -> None:
    spec = WslWorkerLaunchSpec("Ubuntu-ROCm", PurePosixPath("/usr/bin/python3"))
    assert spec.command(PurePosixPath("/temper/request.json")) == (
        "wsl.exe",
        "--distribution",
        "Ubuntu-ROCm",
        "--exec",
        "/usr/bin/python3",
        "-m",
        "temper_ml.runtime.worker_process",
        "--request",
        "/temper/request.json",
    )
    with pytest.raises(WorkerPortError, match="wsl_distribution_invalid"):
        WslWorkerLaunchSpec("Ubuntu && command", PurePosixPath("/usr/bin/python3"))


def test_worker_response_rejects_a_reversed_transfer_manifest() -> None:
    invocation = _invocation()
    manifest = build_transfer_manifest(
        TransferDirection.HOST_TO_WORKER,
        {
            PortableLocation("probes/session/outputs/capability.json"): (
                "capability_profile",
                b"{}",
            )
        },
    )

    with pytest.raises(WorkerPortError, match="worker_response_invalid"):
        WorkerResponse(
            invocation.context,
            "completed",
            manifest,
            {
                "capability_location": {
                    "logical_path": "probes/session/outputs/capability.json"
                }
            },
        )


def test_invocation_round_trip_and_terminal_reconnect_do_not_spawn(
    tmp_path: Path,
) -> None:
    invocation = _invocation()
    assert (
        WorkerInvocation.from_private_bytes(invocation.to_private_bytes()) == invocation
    )
    expected = _terminal_evidence(tmp_path, invocation)
    delivered: list[RuntimeMessage] = []

    def forbidden_popen(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reconnection must not launch a duplicate worker")

    launcher = WslWorkerLauncher(popen=forbidden_popen)
    result = launcher.launch(
        WslWorkerLaunchSpec("Ubuntu-ROCm", PurePosixPath("/usr/bin/python3")),
        invocation,
        invocation_host_path=(tmp_path / "invocation.json").resolve(),
        invocation_worker_path=PurePosixPath("/temper-staging/invocation.json"),
        message_host_root=(tmp_path / "messages").resolve(),
        response_host_path=(tmp_path / "response.json").resolve(),
        cancellation_host_path=(tmp_path / "cancel.json").resolve(),
        interruption_host_path=(tmp_path / "interrupt.json").resolve(),
        on_message=delivered.append,
        cancellation_requested=lambda: False,
        interruption_requested=lambda: False,
    )
    assert result.reused is True
    assert result.response == expected
    assert [message.sequence for message in delivered] == [1, 2, 3]


def test_partial_ledger_requires_reconciliation_instead_of_duplicate_launch(
    tmp_path: Path,
) -> None:
    invocation = _invocation()
    message_root = tmp_path / "messages"
    message_root.mkdir()
    launched = RuntimeMessage(
        IDENTITY,
        invocation.context.run_id,
        1,
        RuntimeMessageKind.LAUNCHED,
        {"operation": "probe", "target_class": "wsl_rocm"},
    )
    (message_root / "00000001.json").write_bytes(launched.to_bytes())
    launcher = WslWorkerLauncher(
        popen=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("partial evidence must not relaunch")
        )
    )

    with pytest.raises(WorkerPortError, match="worker_reconciliation_required"):
        launcher.launch(
            WslWorkerLaunchSpec("Ubuntu-ROCm", PurePosixPath("/usr/bin/python3")),
            invocation,
            invocation_host_path=(tmp_path / "invocation.json").resolve(),
            invocation_worker_path=PurePosixPath("/temper-staging/invocation.json"),
            message_host_root=message_root.resolve(),
            response_host_path=(tmp_path / "response.json").resolve(),
            cancellation_host_path=(tmp_path / "cancel.json").resolve(),
            interruption_host_path=(tmp_path / "interrupt.json").resolve(),
            on_message=lambda message: None,
            cancellation_requested=lambda: False,
            interruption_requested=lambda: False,
        )


def test_live_callback_failure_interrupts_worker_and_preserves_durable_evidence(
    tmp_path: Path,
) -> None:
    invocation = _invocation()

    class Process:
        terminated = False

        def poll(self):
            return 1 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 5
            return 1

    process = Process()

    def popen(*args, **kwargs):
        del args, kwargs
        _terminal_evidence(tmp_path, invocation)
        return process

    def reject_message(message: RuntimeMessage) -> None:
        del message
        raise ValueError("private callback detail")

    with pytest.raises(WorkerPortError, match="worker_message_callback_failed") as exc:
        WslWorkerLauncher(popen=popen).launch(
            WslWorkerLaunchSpec("Ubuntu-ROCm", PurePosixPath("/usr/bin/python3")),
            invocation,
            invocation_host_path=(tmp_path / "invocation.json").resolve(),
            invocation_worker_path=PurePosixPath("/temper-staging/invocation.json"),
            message_host_root=(tmp_path / "messages").resolve(),
            response_host_path=(tmp_path / "response.json").resolve(),
            cancellation_host_path=(tmp_path / "cancel.json").resolve(),
            interruption_host_path=(tmp_path / "interrupt.json").resolve(),
            on_message=reject_message,
            cancellation_requested=lambda: False,
            interruption_requested=lambda: False,
        )

    assert process.terminated is True
    assert [message.sequence for message in exc.value.messages] == [1, 2, 3]
    assert (tmp_path / "interrupt.json").is_file()


def test_new_launch_uses_no_shell_and_validates_durable_terminal_evidence(
    tmp_path: Path,
) -> None:
    invocation = _invocation()
    captured: dict[str, object] = {}

    class Process:
        def poll(self):
            return 0

        def wait(self, timeout):
            assert timeout == 5
            return 0

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        _terminal_evidence(tmp_path, invocation)
        return Process()

    result = WslWorkerLauncher(popen=popen).launch(
        WslWorkerLaunchSpec("Ubuntu-ROCm", PurePosixPath("/usr/bin/python3")),
        invocation,
        invocation_host_path=(tmp_path / "invocation.json").resolve(),
        invocation_worker_path=PurePosixPath("/temper-staging/invocation.json"),
        message_host_root=(tmp_path / "messages").resolve(),
        response_host_path=(tmp_path / "response.json").resolve(),
        cancellation_host_path=(tmp_path / "cancel.json").resolve(),
        interruption_host_path=(tmp_path / "interrupt.json").resolve(),
        on_message=lambda message: None,
        cancellation_requested=lambda: False,
        interruption_requested=lambda: False,
    )
    assert result.reused is False
    assert captured["shell"] is False
    assert captured["command"] == WslWorkerLaunchSpec(
        "Ubuntu-ROCm", PurePosixPath("/usr/bin/python3")
    ).command(PurePosixPath("/temper-staging/invocation.json"))
