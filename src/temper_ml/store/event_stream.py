"""Hash-linked, append-only canonical event streams."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Iterator, TypeVar
from uuid import uuid4

from temper_ml.domain.projections import (
    ContentIdentity,
    HashProjection,
    content_identity,
)
from temper_ml.store.canonical_json import (
    CanonicalJsonError,
    dumps_canonical_json,
    loads_canonical_json,
)

EVENT_PROJECTION = HashProjection("event.envelope", "v1")
EVENT_SCHEMA_VERSION = "v1"
_EVENT_NAME = re.compile(r"^([0-9]{20})-([0-9a-f]{64})\.json$")
_ENVELOPE_FIELDS = {
    "schema_version",
    "projection_version",
    "sequence",
    "predecessor_hash",
    "idempotency_key",
    "event_type",
    "payload",
}
StateT = TypeVar("StateT")


class EventStreamError(RuntimeError):
    """Base error for event stream operations."""


class EventStreamCorrupt(EventStreamError):
    """Raised when an event chain fails verification."""


class EventConflict(EventStreamError):
    """Raised when an idempotency key is reused for different content."""


@dataclass(frozen=True)
class EventRequest:
    idempotency_key: str
    event_type: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        if not isinstance(self.event_type, str) or not self.event_type:
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        dumps_canonical_json(dict(self.payload))

    def canonical_fields(self) -> dict[str, object]:
        return {
            "idempotency_key": self.idempotency_key,
            "event_type": self.event_type,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class StoredEvent:
    sequence: int
    predecessor_hash: str | None
    idempotency_key: str
    event_type: str
    payload: Mapping[str, Any]
    identity: ContentIdentity
    path: Path

    def request_fields(self) -> dict[str, object]:
        return {
            "idempotency_key": self.idempotency_key,
            "event_type": self.event_type,
            "payload": dict(self.payload),
        }


class EventStream:
    """One directory containing a verified immutable event chain."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def append(self, request: EventRequest) -> StoredEvent:
        self.directory.mkdir(parents=True, exist_ok=True)
        with _metadata_lock(self.directory / ".append.lock"):
            events = self.read_verified()
            requested = dumps_canonical_json(request.canonical_fields())
            for event in events:
                if event.idempotency_key != request.idempotency_key:
                    continue
                if dumps_canonical_json(event.request_fields()) == requested:
                    return event
                raise EventConflict(
                    "idempotency key conflicts with existing event: "
                    f"{request.idempotency_key!r}"
                )

            sequence = len(events) + 1
            predecessor = events[-1].identity.value if events else None
            envelope = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "projection_version": EVENT_PROJECTION.version,
                "sequence": sequence,
                "predecessor_hash": predecessor,
                **request.canonical_fields(),
            }
            identity = content_identity(EVENT_PROJECTION, envelope)
            path = self.directory / f"{sequence:020d}-{identity.value}.json"
            if path.exists():
                raise EventStreamCorrupt(f"event path already exists: {path.name}")
            _atomic_write(path, dumps_canonical_json(envelope))
            return StoredEvent(
                sequence,
                predecessor,
                request.idempotency_key,
                request.event_type,
                dict(request.payload),
                identity,
                path,
            )

    def read_verified(self) -> tuple[StoredEvent, ...]:
        if not self.directory.exists():
            return ()
        paths: list[Path] = []
        for path in self.directory.iterdir():
            if path.name == ".append.lock":
                continue
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                continue
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise EventStreamCorrupt(
                    f"unexpected event stream entry: {path.name!r}"
                )
            paths.append(path)
        paths.sort()
        events: list[StoredEvent] = []
        seen_keys: set[str] = set()
        predecessor: str | None = None
        for expected_sequence, path in enumerate(paths, 1):
            match = _EVENT_NAME.fullmatch(path.name)
            if match is None:
                raise EventStreamCorrupt(f"invalid event filename: {path.name!r}")
            filename_sequence = int(match.group(1))
            try:
                raw = path.read_bytes()
                envelope = loads_canonical_json(raw)
            except (OSError, CanonicalJsonError, UnicodeError) as exc:
                raise EventStreamCorrupt(f"invalid event content: {path.name}") from exc
            if not isinstance(envelope, dict):
                raise EventStreamCorrupt(
                    f"event envelope is not an object: {path.name}"
                )
            _validate_envelope(envelope, path.name)
            sequence = envelope["sequence"]
            if sequence != expected_sequence or filename_sequence != sequence:
                raise EventStreamCorrupt(
                    f"event sequence is not contiguous: {path.name}"
                )
            if envelope["predecessor_hash"] != predecessor:
                raise EventStreamCorrupt(
                    f"event predecessor link mismatch: {path.name}"
                )
            key = envelope["idempotency_key"]
            if key in seen_keys:
                raise EventStreamCorrupt(f"duplicate event idempotency key: {key!r}")
            seen_keys.add(key)
            identity = content_identity(EVENT_PROJECTION, envelope)
            if match.group(2) != identity.value:
                raise EventStreamCorrupt(f"event filename hash mismatch: {path.name}")
            if raw != dumps_canonical_json(envelope):
                raise EventStreamCorrupt(f"event is not canonical JSON: {path.name}")
            event = StoredEvent(
                sequence=sequence,
                predecessor_hash=envelope["predecessor_hash"],
                idempotency_key=key,
                event_type=envelope["event_type"],
                payload=envelope["payload"],
                identity=identity,
                path=path,
            )
            events.append(event)
            predecessor = identity.value
        return tuple(events)

    def rebuild(
        self, initial_state: StateT, reducer: Callable[[StateT, StoredEvent], StateT]
    ) -> StateT:
        """Recompute derived state solely from a fully verified event chain."""

        state = initial_state
        for event in self.read_verified():
            state = reducer(state, event)
        return state


def _validate_envelope(envelope: dict[str, Any], name: str) -> None:
    if set(envelope) != _ENVELOPE_FIELDS:
        raise EventStreamCorrupt(f"event envelope fields are invalid: {name}")
    if envelope["schema_version"] != EVENT_SCHEMA_VERSION:
        raise EventStreamCorrupt(f"unsupported event schema version: {name}")
    if envelope["projection_version"] != EVENT_PROJECTION.version:
        raise EventStreamCorrupt(f"unsupported event projection version: {name}")
    sequence = envelope["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise EventStreamCorrupt(f"invalid event sequence: {name}")
    predecessor = envelope["predecessor_hash"]
    if predecessor is not None and (
        not isinstance(predecessor, str)
        or re.fullmatch(r"[0-9a-f]{64}", predecessor) is None
    ):
        raise EventStreamCorrupt(f"invalid event predecessor hash: {name}")
    for field in ("idempotency_key", "event_type"):
        if not isinstance(envelope[field], str) or not envelope[field]:
            raise EventStreamCorrupt(f"invalid event {field}: {name}")
    if not isinstance(envelope["payload"], dict):
        raise EventStreamCorrupt(f"invalid event payload: {name}")


@contextmanager
def _metadata_lock(path: Path) -> Iterator[None]:
    """Hold a one-byte OS lock that the kernel releases when the handle closes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt_api: Any = msvcrt
            msvcrt_api.locking(handle.fileno(), msvcrt_api.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt_api.locking(handle.fileno(), msvcrt_api.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl_api: Any = fcntl
            fcntl_api.flock(handle.fileno(), fcntl_api.LOCK_EX)
            try:
                yield
            finally:
                fcntl_api.flock(handle.fileno(), fcntl_api.LOCK_UN)


def _atomic_write(path: Path, payload: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
