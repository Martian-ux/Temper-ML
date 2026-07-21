"""Cross-process ownership for one local run launch attempt."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import stat
from typing import Any, Iterator

from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import (
    RecordValidationError,
    identity_fields,
    parse_identity,
    require_identifier,
)
from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    ensure_safe_directory,
    is_link_or_reparse,
    require_safe_directory,
    require_safe_regular_file,
    safe_path_stat,
    same_file_object,
)
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json
from temper_ml.store.safe_io import SafeIoError, read_stable_bytes, write_once_bytes


class RunOwnershipError(RuntimeError):
    """A stable ownership failure without process or machine details."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass
class RunOwnershipLease:
    """A held OS lease whose durable claim is released only by resolution."""

    resolution_path: Path
    resolution_payload: bytes = field(repr=False)
    _resolved: bool = field(default=False, init=False, repr=False)

    def resolve(self) -> None:
        """Durably record that the launch never started or reached a terminal event."""

        try:
            try:
                write_once_bytes(self.resolution_path, self.resolution_payload)
            except FileExistsError:
                if read_stable_bytes(self.resolution_path) != self.resolution_payload:
                    raise RunOwnershipError("run_ownership_resolution_conflict")
        except RunOwnershipError:
            raise
        except (OSError, SafeIoError, UnsafeFilesystemPath):
            try:
                if read_stable_bytes(self.resolution_path) != self.resolution_payload:
                    raise RunOwnershipError("run_ownership_resolution_conflict")
            except RunOwnershipError:
                raise
            except (OSError, SafeIoError, UnsafeFilesystemPath):
                raise RunOwnershipError("run_ownership_resolution_failed") from None
        self._resolved = True


def released_run_claim_identity(
    private_root: Path,
    run_id: str,
) -> ContentIdentity:
    """Read one exact durable release without creating ownership state."""

    claim_identity = existing_run_claim_identity(private_root, run_id)
    root = private_root / "runtime-ownership" / run_id
    resolution_path = root / "resolved.json"
    try:
        if safe_path_stat(resolution_path, allow_missing=True) is None:
            raise RunOwnershipError("run_ownership_unresolved")
        resolution_bytes = read_stable_bytes(resolution_path)
        resolution = loads_canonical_json(resolution_bytes)
        expected_resolution = {
            "schema_version": "v1",
            "run_id": run_id,
            "claim_identity": identity_fields(claim_identity),
            "resolved": True,
        }
        if resolution != expected_resolution:
            raise RunOwnershipError("run_ownership_resolution_conflict")
        if resolution_bytes != dumps_canonical_json(expected_resolution):
            raise RunOwnershipError("run_ownership_resolution_conflict")
        return claim_identity
    except RunOwnershipError:
        raise
    except (
        OSError,
        SafeIoError,
        UnsafeFilesystemPath,
        RecordValidationError,
        TypeError,
        ValueError,
    ):
        raise RunOwnershipError("run_ownership_resolution_conflict") from None


def existing_run_claim_identity(
    private_root: Path,
    run_id: str,
) -> ContentIdentity:
    """Read and validate one immutable claim whether or not it is resolved."""

    if not isinstance(private_root, Path) or not private_root.is_absolute():
        raise RunOwnershipError("run_ownership_root_invalid")
    try:
        require_identifier("run_id", run_id)
    except RecordValidationError:
        raise RunOwnershipError("run_ownership_subject_invalid") from None
    claim_path = private_root / "runtime-ownership" / run_id / "claim.json"
    try:
        claim_bytes = read_stable_bytes(claim_path)
        claim = loads_canonical_json(claim_bytes)
        if not isinstance(claim, dict) or set(claim) != {
            "schema_version",
            "run_id",
            "claim_identity",
        }:
            raise RunOwnershipError("run_ownership_claim_conflict")
        if claim.get("schema_version") != "v1" or claim.get("run_id") != run_id:
            raise RunOwnershipError("run_ownership_claim_conflict")
        raw_identity = claim.get("claim_identity")
        if not isinstance(raw_identity, dict):
            raise RunOwnershipError("run_ownership_claim_conflict")
        claim_identity = parse_identity(raw_identity, field="run ownership claim")
        if claim_bytes != dumps_canonical_json(claim):
            raise RunOwnershipError("run_ownership_claim_conflict")
        return claim_identity
    except RunOwnershipError:
        raise
    except FileNotFoundError:
        raise RunOwnershipError("run_ownership_claim_missing") from None
    except (
        OSError,
        SafeIoError,
        UnsafeFilesystemPath,
        RecordValidationError,
        TypeError,
        ValueError,
    ):
        raise RunOwnershipError("run_ownership_claim_conflict") from None


def reconcile_run_ownership(
    private_root: Path,
    run_id: str,
    claim_identity: ContentIdentity,
) -> ContentIdentity:
    """Resolve one exact existing claim after its caller verifies run terminality."""

    if existing_run_claim_identity(private_root, run_id) != claim_identity:
        raise RunOwnershipError("run_ownership_claim_conflict")
    with _claim_run_ownership(
        private_root,
        run_id,
        claim_identity,
        allow_create=False,
        require_resolved=False,
    ) as lease:
        lease.resolve()
    released = released_run_claim_identity(private_root, run_id)
    if released != claim_identity:
        raise RunOwnershipError("run_ownership_resolution_conflict")
    return released


@contextmanager
def claim_run_ownership(
    private_root: Path,
    run_id: str,
    claim_identity: ContentIdentity,
) -> Iterator[RunOwnershipLease]:
    """Hold one non-blocking OS lease bound to a durable immutable claim."""

    with _claim_run_ownership(
        private_root,
        run_id,
        claim_identity,
        allow_create=True,
        require_resolved=True,
    ) as lease:
        yield lease


@contextmanager
def claim_released_run_ownership(
    private_root: Path,
    run_id: str,
    claim_identity: ContentIdentity,
) -> Iterator[RunOwnershipLease]:
    """Hold one existing released run lease without creating control bytes."""

    with _claim_run_ownership(
        private_root,
        run_id,
        claim_identity,
        allow_create=False,
        require_resolved=True,
    ) as lease:
        yield lease


@contextmanager
def claim_abandoned_run_ownership(
    private_root: Path,
    run_id: str,
    claim_identity: ContentIdentity,
) -> Iterator[RunOwnershipLease]:
    """Acquire one exact unresolved claim without creating ownership bytes."""

    with _claim_run_ownership(
        private_root,
        run_id,
        claim_identity,
        allow_create=False,
        require_resolved=False,
        require_unresolved=True,
    ) as lease:
        yield lease


@contextmanager
def _claim_run_ownership(
    private_root: Path,
    run_id: str,
    claim_identity: ContentIdentity,
    *,
    allow_create: bool,
    require_resolved: bool,
    require_unresolved: bool = False,
) -> Iterator[RunOwnershipLease]:
    """Implement creating launch claims and non-creating released claims."""

    if require_resolved and require_unresolved:
        raise RunOwnershipError("run_ownership_state_invalid")
    if not isinstance(private_root, Path) or not private_root.is_absolute():
        raise RunOwnershipError("run_ownership_root_invalid")
    try:
        require_identifier("run_id", run_id)
    except RecordValidationError:
        raise RunOwnershipError("run_ownership_subject_invalid") from None
    if not isinstance(claim_identity, ContentIdentity):
        raise RunOwnershipError("run_ownership_subject_invalid")
    root = private_root / "runtime-ownership" / run_id
    claim_path = root / "claim.json"
    lock_path = root / "lease.lock"
    resolution_path = root / "resolved.json"
    claim = dumps_canonical_json(
        {
            "schema_version": "v1",
            "run_id": run_id,
            "claim_identity": identity_fields(claim_identity),
        }
    )
    resolution = dumps_canonical_json(
        {
            "schema_version": "v1",
            "run_id": run_id,
            "claim_identity": identity_fields(claim_identity),
            "resolved": True,
        }
    )
    claim_created = False
    try:
        if allow_create:
            ensure_safe_directory(root)
            try:
                write_once_bytes(claim_path, claim)
                claim_created = True
            except FileExistsError:
                if read_stable_bytes(claim_path) != claim:
                    raise RunOwnershipError("run_ownership_claim_conflict")
        else:
            require_safe_directory(root)
            if read_stable_bytes(claim_path) != claim:
                raise RunOwnershipError("run_ownership_claim_conflict")
        existing = safe_path_stat(lock_path, allow_missing=True)
        if existing is None and not allow_create:
            raise RunOwnershipError("run_ownership_path_invalid")
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise RunOwnershipError("run_ownership_path_invalid")
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if allow_create:
            flags |= os.O_CREAT
        descriptor = os.open(lock_path, flags, 0o600)
    except RunOwnershipError:
        raise
    except (OSError, SafeIoError, UnsafeFilesystemPath):
        raise RunOwnershipError("run_ownership_unavailable") from None
    with os.fdopen(descriptor, "r+b") as handle:
        try:
            opened = os.fstat(handle.fileno())
            current = require_safe_regular_file(lock_path)
            if (
                is_link_or_reparse(opened)
                or not stat.S_ISREG(opened.st_mode)
                or not same_file_object(opened, current)
            ):
                raise RunOwnershipError("run_ownership_path_invalid")
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0 and allow_create:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            elif size != 1:
                raise RunOwnershipError("run_ownership_path_invalid")
            handle.seek(0)
            _lock_nonblocking(handle)
            handle.seek(0)
            if handle.read(1) != b"\0":
                raise RunOwnershipError("run_ownership_path_invalid")
            existing_resolution = safe_path_stat(resolution_path, allow_missing=True)
            if claim_created:
                if existing_resolution is not None:
                    raise RunOwnershipError("run_ownership_resolution_conflict")
            elif existing_resolution is None:
                if require_resolved:
                    raise RunOwnershipError("run_ownership_unresolved")
            else:
                if read_stable_bytes(resolution_path) != resolution:
                    raise RunOwnershipError("run_ownership_resolution_conflict")
                if require_unresolved:
                    raise RunOwnershipError("run_ownership_resolved")
        except RunOwnershipError:
            raise
        except (OSError, SafeIoError, UnsafeFilesystemPath):
            raise RunOwnershipError("run_ownership_unavailable") from None
        lease = RunOwnershipLease(resolution_path, resolution)
        try:
            yield lease
        finally:
            try:
                handle.seek(0)
                _unlock(handle)
            except OSError:
                pass


def _lock_nonblocking(handle: Any) -> None:
    try:
        if os.name == "nt":
            msvcrt: Any = importlib.import_module("msvcrt")
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise RunOwnershipError("run_ownership_unavailable") from None


def _unlock(handle: Any) -> None:
    if os.name == "nt":
        msvcrt: Any = importlib.import_module("msvcrt")
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl: Any = importlib.import_module("fcntl")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
