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
    require_identifier,
)
from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    ensure_safe_directory,
    is_link_or_reparse,
    require_safe_regular_file,
    safe_path_stat,
    same_file_object,
)
from temper_ml.store.canonical_json import dumps_canonical_json
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
            raise RunOwnershipError("run_ownership_resolution_failed") from None
        self._resolved = True


@contextmanager
def claim_run_ownership(
    private_root: Path,
    run_id: str,
    claim_identity: ContentIdentity,
) -> Iterator[RunOwnershipLease]:
    """Hold one non-blocking OS lease bound to a durable immutable claim."""

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
        ensure_safe_directory(root)
        try:
            write_once_bytes(claim_path, claim)
            claim_created = True
        except FileExistsError:
            if read_stable_bytes(claim_path) != claim:
                raise RunOwnershipError("run_ownership_claim_conflict")
        existing = safe_path_stat(lock_path, allow_missing=True)
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise RunOwnershipError("run_ownership_path_invalid")
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
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
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            _lock_nonblocking(handle)
            existing_resolution = safe_path_stat(resolution_path, allow_missing=True)
            if claim_created:
                if existing_resolution is not None:
                    raise RunOwnershipError("run_ownership_resolution_conflict")
            elif existing_resolution is None:
                raise RunOwnershipError("run_ownership_unresolved")
            elif read_stable_bytes(resolution_path) != resolution:
                raise RunOwnershipError("run_ownership_resolution_conflict")
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
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise RunOwnershipError("run_ownership_unavailable") from None


def _unlock(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl: Any = importlib.import_module("fcntl")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
