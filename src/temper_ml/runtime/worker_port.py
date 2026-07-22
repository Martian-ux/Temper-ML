"""Private process port for one local Windows-hosted WSL worker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import os
import re
import subprocess
import time
from typing import Any

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.domain.recipes import RecipeResolution
from temper_ml.domain.records import (
    RecordEnvelope,
    RecordValidationError,
    freeze_json_object,
    identity_fields,
    parse_identity,
    thaw_json,
)
from temper_ml.runtime.controller import (
    ControllerState,
    RuntimeControllerError,
    SerializedRunController,
)
from temper_ml.runtime.fixture_inference import (
    FixtureInferenceError,
    InferenceSettings,
)
from temper_ml.runtime.library_backend import LibraryExecutionContext
from temper_ml.runtime.paths import PortableLocation, PortablePathError
from temper_ml.runtime.protocol import (
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperation,
    RuntimeProtocolError,
)
from temper_ml.runtime.staging import (
    StagingError,
    TransferDirection,
    TransferManifest,
)
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes, write_once_bytes

WORKER_RESPONSE_PROJECTION = HashProjection("runtime.worker_response", "v1")
WORKER_MODULE = "temper_ml.runtime.worker_process"
_DISTRIBUTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class WorkerPortError(RuntimeError):
    """One stable process-boundary failure without private diagnostics."""

    def __init__(
        self,
        code: str,
        *,
        messages: tuple[RuntimeMessage, ...] = (),
        response: "WorkerResponse | None" = None,
    ) -> None:
        self.code = code
        self.messages = messages
        self.response = response
        super().__init__(code)


@dataclass(frozen=True)
class WorkerInvocation:
    """Immutable private request consumed by exactly one local worker process."""

    context: LibraryExecutionContext
    worker_root: PurePosixPath
    output_prefix: PortableLocation
    model_source: PurePosixPath | None = None
    tokenizer_source: PurePosixPath | None = None
    resolution: RecipeResolution | None = None
    input_manifest: TransferManifest | None = None
    settings: InferenceSettings | None = None
    adapter_payload_format: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, LibraryExecutionContext):
            raise WorkerPortError("worker_invocation_invalid")
        if (
            not isinstance(self.worker_root, PurePosixPath)
            or not self.worker_root.is_absolute()
        ):
            raise WorkerPortError("worker_invocation_invalid")
        if not isinstance(self.output_prefix, PortableLocation):
            raise WorkerPortError("worker_invocation_invalid")
        for source in (self.model_source, self.tokenizer_source):
            if source is not None and (
                not isinstance(source, PurePosixPath) or not source.is_absolute()
            ):
                raise WorkerPortError("worker_invocation_invalid")
        operation = self.context.operation
        if operation is RuntimeOperation.PROBE:
            if any(
                value is not None
                for value in (
                    self.model_source,
                    self.tokenizer_source,
                    self.resolution,
                    self.input_manifest,
                    self.settings,
                    self.adapter_payload_format,
                )
            ):
                raise WorkerPortError("worker_invocation_invalid")
            return
        if (
            self.model_source is None
            or self.tokenizer_source is None
            or not isinstance(self.resolution, RecipeResolution)
            or not isinstance(self.input_manifest, TransferManifest)
            or self.input_manifest.direction is not TransferDirection.HOST_TO_WORKER
        ):
            raise WorkerPortError("worker_invocation_invalid")
        if operation is RuntimeOperation.TRAIN:
            if self.settings is not None or self.adapter_payload_format is not None:
                raise WorkerPortError("worker_invocation_invalid")
            return
        if operation not in {
            RuntimeOperation.EVALUATE,
            RuntimeOperation.INFER_FOCUSED,
            RuntimeOperation.INFER_BATCH,
        }:
            raise WorkerPortError("worker_invocation_invalid")
        if not isinstance(self.settings, InferenceSettings) or (
            self.adapter_payload_format != "safetensors"
        ):
            raise WorkerPortError("worker_invocation_invalid")

    @property
    def message_prefix(self) -> PortableLocation:
        return PortableLocation(f"{self.output_prefix.logical_path}/messages")

    @property
    def response_location(self) -> PortableLocation:
        return PortableLocation(f"{self.output_prefix.logical_path}/response.json")

    @property
    def cancellation_location(self) -> PortableLocation:
        return PortableLocation(f"{self.output_prefix.logical_path}/cancel.json")

    @property
    def interruption_location(self) -> PortableLocation:
        return PortableLocation(f"{self.output_prefix.logical_path}/interrupt.json")

    def to_private_dict(self) -> dict[str, object]:
        return {
            "schema_version": "v1",
            "request_identity": identity_fields(self.context.request_identity),
            "run_id": self.context.run_id,
            "operation": self.context.operation.value,
            "target_class": self.context.target_class,
            "worker_root": self.worker_root.as_posix(),
            "output_prefix": self.output_prefix.to_dict(),
            "model_source": (
                self.model_source.as_posix() if self.model_source is not None else None
            ),
            "tokenizer_source": (
                self.tokenizer_source.as_posix()
                if self.tokenizer_source is not None
                else None
            ),
            "resolution": (
                self.resolution.to_dict() if self.resolution is not None else None
            ),
            "input_manifest": (
                self.input_manifest.to_dict()
                if self.input_manifest is not None
                else None
            ),
            "settings": self.settings.to_dict() if self.settings is not None else None,
            "adapter_payload_format": self.adapter_payload_format,
        }

    def to_private_bytes(self) -> bytes:
        return dumps_canonical_json(self.to_private_dict())

    @classmethod
    def from_private_bytes(cls, data: bytes) -> "WorkerInvocation":
        try:
            value = loads_canonical_json(data)
        except (CanonicalJsonError, TypeError, ValueError):
            raise WorkerPortError("worker_invocation_invalid") from None
        expected = {
            "schema_version",
            "request_identity",
            "run_id",
            "operation",
            "target_class",
            "worker_root",
            "output_prefix",
            "model_source",
            "tokenizer_source",
            "resolution",
            "input_manifest",
            "settings",
            "adapter_payload_format",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value["schema_version"] != "v1"
            or dumps_canonical_json(value) != data
        ):
            raise WorkerPortError("worker_invocation_invalid")
        try:
            raw_prefix = value["output_prefix"]
            if not isinstance(raw_prefix, Mapping) or set(raw_prefix) != {
                "logical_path"
            }:
                raise WorkerPortError("worker_invocation_invalid")
            context = LibraryExecutionContext(
                parse_identity(value["request_identity"], field="request_identity"),
                value["run_id"],
                RuntimeOperation(value["operation"]),
                value["target_class"],
            )
            resolution = _optional_resolution(value["resolution"])
            manifest = _optional_manifest(value["input_manifest"])
            settings = _optional_settings(value["settings"])
            return cls(
                context=context,
                worker_root=PurePosixPath(value["worker_root"]),
                output_prefix=PortableLocation(raw_prefix["logical_path"]),
                model_source=_optional_posix_path(value["model_source"]),
                tokenizer_source=_optional_posix_path(value["tokenizer_source"]),
                resolution=resolution,
                input_manifest=manifest,
                settings=settings,
                adapter_payload_format=value["adapter_payload_format"],
            )
        except WorkerPortError:
            raise
        except (
            FixtureInferenceError,
            PortablePathError,
            RecordValidationError,
            StagingError,
            TypeError,
            ValueError,
        ):
            raise WorkerPortError("worker_invocation_invalid") from None


@dataclass(frozen=True)
class WorkerResponse:
    """Subject-bound worker result; all referenced bytes remain separately verified."""

    context: LibraryExecutionContext
    status: str
    output_manifest: TransferManifest | None
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, LibraryExecutionContext):
            raise WorkerPortError("worker_response_invalid")
        if self.status not in {"completed", "cancelled", "interrupted", "failed"}:
            raise WorkerPortError("worker_response_invalid")
        if (self.status == "completed") != isinstance(
            self.output_manifest, TransferManifest
        ) and not (
            self.status == "interrupted"
            and isinstance(self.output_manifest, TransferManifest)
        ):
            raise WorkerPortError("worker_response_invalid")
        if (
            self.output_manifest is not None
            and self.output_manifest.direction is not TransferDirection.WORKER_TO_HOST
        ):
            raise WorkerPortError("worker_response_invalid")
        try:
            frozen = freeze_json_object(self.payload, field="worker_response.payload")
        except (RecordValidationError, TypeError, ValueError):
            raise WorkerPortError("worker_response_invalid") from None
        object.__setattr__(self, "payload", frozen)

    @property
    def identity(self) -> ContentIdentity:
        return content_identity(WORKER_RESPONSE_PROJECTION, self.projected_fields())

    def projected_fields(self) -> dict[str, object]:
        return {
            "schema_version": "v1",
            "request_identity": identity_fields(self.context.request_identity),
            "run_id": self.context.run_id,
            "operation": self.context.operation.value,
            "target_class": self.context.target_class,
            "status": self.status,
            "output_manifest": (
                self.output_manifest.to_dict()
                if self.output_manifest is not None
                else None
            ),
            "payload": thaw_json(self.payload),
        }

    def to_dict(self) -> dict[str, object]:
        value = self.projected_fields()
        value["identity"] = identity_fields(self.identity)
        return value

    def to_bytes(self) -> bytes:
        return dumps_canonical_json(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "WorkerResponse":
        try:
            value = loads_canonical_json(data)
        except (CanonicalJsonError, TypeError, ValueError):
            raise WorkerPortError("worker_response_invalid") from None
        expected = {
            "schema_version",
            "request_identity",
            "run_id",
            "operation",
            "target_class",
            "status",
            "output_manifest",
            "payload",
            "identity",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value["schema_version"] != "v1"
            or dumps_canonical_json(value) != data
        ):
            raise WorkerPortError("worker_response_invalid")
        try:
            context = LibraryExecutionContext(
                parse_identity(value["request_identity"], field="request_identity"),
                value["run_id"],
                RuntimeOperation(value["operation"]),
                value["target_class"],
            )
            response = cls(
                context,
                value["status"],
                _optional_manifest(value["output_manifest"]),
                value["payload"],
            )
            claimed = parse_identity(value["identity"], field="identity")
        except WorkerPortError:
            raise
        except (RecordValidationError, StagingError, TypeError, ValueError):
            raise WorkerPortError("worker_response_invalid") from None
        if response.identity != claimed:
            raise WorkerPortError("worker_response_identity_mismatch")
        return response


@dataclass(frozen=True)
class WslWorkerLaunchSpec:
    distribution: str
    python_executable: PurePosixPath
    timeout_seconds: int = 3600
    poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, str) or not _DISTRIBUTION.fullmatch(
            self.distribution
        ):
            raise WorkerPortError("wsl_distribution_invalid")
        if (
            not isinstance(self.python_executable, PurePosixPath)
            or not self.python_executable.is_absolute()
        ):
            raise WorkerPortError("wsl_python_invalid")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds < 1
            or isinstance(self.poll_interval_seconds, bool)
            or not isinstance(self.poll_interval_seconds, (int, float))
            or not 0 < self.poll_interval_seconds <= 1
        ):
            raise WorkerPortError("worker_timeout_invalid")

    def command(self, invocation_path: PurePosixPath) -> tuple[str, ...]:
        if (
            not isinstance(invocation_path, PurePosixPath)
            or not invocation_path.is_absolute()
        ):
            raise WorkerPortError("worker_invocation_path_invalid")
        return (
            "wsl.exe",
            "--distribution",
            self.distribution,
            "--exec",
            self.python_executable.as_posix(),
            "-m",
            WORKER_MODULE,
            "--request",
            invocation_path.as_posix(),
        )


@dataclass(frozen=True)
class WorkerLaunchResult:
    response: WorkerResponse
    messages: tuple[RuntimeMessage, ...]
    reused: bool


MessageCallback = Callable[[RuntimeMessage], None]
ControlCheck = Callable[[], bool]


class WslWorkerLauncher:
    """Launch once, monitor durable messages, and reuse exact terminal evidence."""

    def __init__(
        self,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        self._popen = popen
        self._monotonic = monotonic
        self._wait = wait

    def launch(
        self,
        spec: WslWorkerLaunchSpec,
        invocation: WorkerInvocation,
        *,
        invocation_host_path: Path,
        invocation_worker_path: PurePosixPath,
        message_host_root: Path,
        response_host_path: Path,
        cancellation_host_path: Path,
        interruption_host_path: Path,
        on_message: MessageCallback,
        cancellation_requested: ControlCheck,
        interruption_requested: ControlCheck,
    ) -> WorkerLaunchResult:
        if not isinstance(spec, WslWorkerLaunchSpec) or not isinstance(
            invocation, WorkerInvocation
        ):
            raise WorkerPortError("worker_launch_invalid")
        for path in (
            invocation_host_path,
            message_host_root,
            response_host_path,
            cancellation_host_path,
            interruption_host_path,
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise WorkerPortError("worker_launch_invalid")
        if (
            not callable(on_message)
            or not callable(cancellation_requested)
            or not callable(interruption_requested)
        ):
            raise WorkerPortError("worker_launch_invalid")
        _write_idempotent(invocation_host_path, invocation.to_private_bytes())
        existing = _read_messages(message_host_root, invocation.context)
        if response_host_path.is_file():
            existing_response = _read_response(response_host_path, invocation.context)
            _validate_terminal(existing_response, existing)
            for message in existing:
                _deliver_message(on_message, message, existing)
            return WorkerLaunchResult(existing_response, existing, True)
        if existing:
            raise WorkerPortError("worker_reconciliation_required", messages=existing)

        creation_flags = (
            int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        )
        try:
            process = self._popen(
                spec.command(invocation_worker_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=creation_flags,
            )
        except (OSError, ValueError):
            raise WorkerPortError("worker_launch_failed") from None
        delivered = 0
        started = self._monotonic()
        timed_out = False
        try:
            while process.poll() is None:
                messages = _read_messages(message_host_root, invocation.context)
                for message in messages[delivered:]:
                    _deliver_message(on_message, message, messages)
                delivered = len(messages)
                if cancellation_requested():
                    _write_control_marker(cancellation_host_path, "cancel")
                if interruption_requested():
                    _write_control_marker(interruption_host_path, "interrupt")
                if self._monotonic() - started >= spec.timeout_seconds:
                    timed_out = True
                    _write_control_marker(interruption_host_path, "interrupt")
                    break
                self._wait(float(spec.poll_interval_seconds))
        except Exception:
            try:
                _write_control_marker(interruption_host_path, "interrupt")
            except (OSError, WorkerPortError):
                pass
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise
        if timed_out:
            deadline = self._monotonic() + min(5.0, spec.timeout_seconds)
            while process.poll() is None and self._monotonic() < deadline:
                self._wait(float(spec.poll_interval_seconds))
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        try:
            return_code = process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return_code = None
        messages = _read_messages(message_host_root, invocation.context)
        for message in messages[delivered:]:
            _deliver_message(on_message, message, messages)
        response = (
            _read_response(response_host_path, invocation.context)
            if response_host_path.is_file()
            else None
        )
        if timed_out:
            raise WorkerPortError(
                "worker_timeout", messages=messages, response=response
            )
        if return_code != 0 or response is None:
            raise WorkerPortError(
                "worker_disconnected", messages=messages, response=response
            )
        _validate_terminal(response, messages)
        return WorkerLaunchResult(response, messages, False)


def _deliver_message(
    callback: MessageCallback,
    message: RuntimeMessage,
    durable_messages: tuple[RuntimeMessage, ...],
) -> None:
    try:
        callback(message)
    except Exception:
        raise WorkerPortError(
            "worker_message_callback_failed", messages=durable_messages
        ) from None


class WorkerEventSink:
    """Worker-side durable ordered message writer with one serialized owner."""

    def __init__(self, invocation: WorkerInvocation) -> None:
        if not isinstance(invocation, WorkerInvocation):
            raise WorkerPortError("worker_event_sink_invalid")
        self.invocation = invocation
        self.controller = SerializedRunController(
            invocation.context.request_identity, invocation.context.run_id
        )
        self.sequence = 0

    def emit(
        self, kind: RuntimeMessageKind, payload: Mapping[str, object]
    ) -> RuntimeMessage:
        self.sequence += 1
        message = RuntimeMessage(
            self.invocation.context.request_identity,
            self.invocation.context.run_id,
            self.sequence,
            kind,
            payload,
        )
        try:
            self.controller.accept(message)
        except RuntimeControllerError:
            raise WorkerPortError("worker_event_transition_invalid") from None
        path = (
            Path(self.invocation.worker_root.as_posix())
            / self.invocation.message_prefix.logical_path
            / f"{self.sequence:08d}.json"
        )
        _write_idempotent(path, message.to_bytes())
        return message


def control_requested(invocation: WorkerInvocation, kind: str) -> bool:
    if kind not in {"cancel", "interrupt"}:
        raise WorkerPortError("worker_control_invalid")
    location = (
        invocation.cancellation_location
        if kind == "cancel"
        else invocation.interruption_location
    )
    path = Path(invocation.worker_root.as_posix()) / location.logical_path
    if not path.exists():
        return False
    try:
        value = loads_canonical_json(read_stable_bytes(path))
    except (CanonicalJsonError, SafeIoError, TypeError, ValueError):
        raise WorkerPortError("worker_control_invalid") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "control"}
        or value["schema_version"] != "v1"
        or value["control"] not in {kind, "timeout"}
    ):
        raise WorkerPortError("worker_control_invalid")
    return True


def write_worker_response(
    invocation: WorkerInvocation, response: WorkerResponse
) -> None:
    if response.context != invocation.context:
        raise WorkerPortError("worker_response_subject_mismatch")
    path = (
        Path(invocation.worker_root.as_posix())
        / invocation.response_location.logical_path
    )
    _write_idempotent(path, response.to_bytes())


def _optional_posix_path(value: object) -> PurePosixPath | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkerPortError("worker_invocation_invalid")
    return PurePosixPath(value)


def _optional_resolution(value: object) -> RecipeResolution | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WorkerPortError("worker_invocation_invalid")
    record = RecordEnvelope.from_dict(value).to_record()
    if not isinstance(record, RecipeResolution):
        raise WorkerPortError("worker_invocation_invalid")
    return record


def _optional_manifest(value: object) -> TransferManifest | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WorkerPortError("worker_manifest_invalid")
    return TransferManifest.from_dict(value)


def _optional_settings(value: object) -> InferenceSettings | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WorkerPortError("worker_invocation_invalid")
    return InferenceSettings.from_mapping(value)


def _read_messages(
    root: Path, context: LibraryExecutionContext
) -> tuple[RuntimeMessage, ...]:
    if not root.exists():
        return ()
    if not root.is_dir():
        raise WorkerPortError("worker_message_ledger_invalid")
    messages: list[RuntimeMessage] = []
    try:
        paths = tuple(sorted(root.glob("*.json"), key=lambda path: path.name))
    except OSError:
        raise WorkerPortError("worker_message_ledger_invalid") from None
    for sequence, path in enumerate(paths, 1):
        if path.name != f"{sequence:08d}.json":
            raise WorkerPortError("worker_message_ledger_invalid")
        try:
            message = RuntimeMessage.from_bytes(read_stable_bytes(path))
        except (RuntimeProtocolError, SafeIoError):
            raise WorkerPortError("worker_message_ledger_invalid") from None
        if (
            message.request_identity != context.request_identity
            or message.run_id != context.run_id
        ):
            raise WorkerPortError("worker_message_subject_mismatch")
        messages.append(message)
    try:
        SerializedRunController.reconstruct(
            context.request_identity, context.run_id, messages
        )
    except RuntimeControllerError:
        raise WorkerPortError("worker_message_ledger_invalid") from None
    return tuple(messages)


def _read_response(path: Path, context: LibraryExecutionContext) -> WorkerResponse:
    try:
        response = WorkerResponse.from_bytes(read_stable_bytes(path))
    except SafeIoError:
        raise WorkerPortError("worker_response_unavailable") from None
    if response.context != context:
        raise WorkerPortError("worker_response_subject_mismatch")
    return response


def _validate_terminal(
    response: WorkerResponse, messages: tuple[RuntimeMessage, ...]
) -> None:
    if not messages:
        raise WorkerPortError("worker_terminal_evidence_missing")
    try:
        snapshot = SerializedRunController.reconstruct(
            response.context.request_identity,
            response.context.run_id,
            messages,
        ).snapshot()
    except RuntimeControllerError:
        raise WorkerPortError("worker_terminal_evidence_invalid") from None
    expected = {
        "completed": ControllerState.COMPLETED,
        "cancelled": ControllerState.CANCELLED,
        "interrupted": ControllerState.INTERRUPTED,
        "failed": ControllerState.FAILED,
    }[response.status]
    if snapshot.state is not expected:
        raise WorkerPortError("worker_terminal_evidence_invalid")
    if response.output_manifest is not None and (
        response.status == "completed"
        and snapshot.artifact_identity != response.output_manifest.identity
    ):
        raise WorkerPortError("worker_terminal_evidence_invalid")


def _write_control_marker(path: Path, control: str) -> None:
    data = dumps_canonical_json({"schema_version": "v1", "control": control})
    _write_idempotent(path, data)


def _write_idempotent(path: Path, data: bytes) -> None:
    try:
        write_once_bytes(path, data)
    except FileExistsError:
        try:
            existing = read_stable_bytes(path)
        except SafeIoError:
            raise WorkerPortError("worker_private_state_unavailable") from None
        if existing != data:
            raise WorkerPortError("worker_private_state_conflict")
    except SafeIoError:
        raise WorkerPortError("worker_private_state_unavailable") from None
