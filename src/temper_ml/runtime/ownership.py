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


_OWNERSHIP_CONTROL_ENTRIES = frozenset({".claims.lock", ".control"})
_LEGACY_CONTROL_DIRECTORIES = frozenset({"cleanup", "cleanup-quarantine", "replay"})


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


@dataclass(frozen=True)
class RunOwnershipClaim:
    """One validated immutable run claim discovered during startup recovery."""

    run_id: str
    claim_identity: ContentIdentity
    resolved: bool
    request_id: str | None = None
    artifact_id: str | None = None
    attempt_number: int | None = None


def iter_run_ownership_claims(private_root: Path) -> tuple[RunOwnershipClaim, ...]:
    """Enumerate only validated run claims, excluding fixed control namespaces."""

    if not isinstance(private_root, Path) or not private_root.is_absolute():
        raise RunOwnershipError("run_ownership_root_invalid")
    ownership_root = private_root / "runtime-ownership"
    try:
        root_info = safe_path_stat(ownership_root, allow_missing=True)
        if root_info is None:
            return ()
        if not stat.S_ISDIR(root_info.st_mode):
            raise RunOwnershipError("run_ownership_path_invalid")
        require_safe_directory(ownership_root)
        with _claim_ownership_namespace(private_root):
            with os.scandir(ownership_root) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
            claims: list[RunOwnershipClaim] = []
            for entry in entries:
                if entry.name in _OWNERSHIP_CONTROL_ENTRIES:
                    continue
                info = entry.stat(follow_symlinks=False)
                if is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
                    raise RunOwnershipError("run_ownership_path_invalid")
                try:
                    run_id = require_identifier("run_id", entry.name)
                except RecordValidationError:
                    raise RunOwnershipError("run_ownership_path_invalid") from None
                run_root = Path(entry.path)
                claim_path = run_root / "claim.json"
                if safe_path_stat(claim_path, allow_missing=True) is None:
                    if entry.name in _LEGACY_CONTROL_DIRECTORIES:
                        continue
                    _validate_unpublished_claim_directory(run_root)
                    continue
                (
                    claim_identity,
                    request_id,
                    artifact_id,
                    attempt_number,
                ) = _existing_run_claim(private_root, run_id)
                _repair_unpublished_run_lock(run_root)
                try:
                    released_run_claim_identity(private_root, run_id)
                    resolved = True
                except RunOwnershipError as exc:
                    if exc.code != "run_ownership_unresolved":
                        raise
                    resolved = False
                claims.append(
                    RunOwnershipClaim(
                        run_id,
                        claim_identity,
                        resolved,
                        request_id,
                        artifact_id,
                        attempt_number,
                    )
                )
            return tuple(claims)
    except RunOwnershipError:
        raise
    except (OSError, SafeIoError, UnsafeFilesystemPath):
        raise RunOwnershipError("run_ownership_path_invalid") from None


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

    return _existing_run_claim(private_root, run_id)[0]


def _existing_run_claim(
    private_root: Path,
    run_id: str,
) -> tuple[ContentIdentity, str | None, str | None, int | None]:
    """Read one v1 identity-only or v2 recovery-describing claim."""

    if not isinstance(private_root, Path) or not private_root.is_absolute():
        raise RunOwnershipError("run_ownership_root_invalid")
    try:
        require_identifier("run_id", run_id)
    except RecordValidationError:
        raise RunOwnershipError("run_ownership_subject_invalid") from None
    claim_path = private_root / "runtime-ownership" / run_id / "claim.json"
    return _read_run_claim_path(claim_path, run_id)


def _read_run_claim_path(
    claim_path: Path,
    run_id: str,
) -> tuple[ContentIdentity, str | None, str | None, int | None]:
    """Parse one canonical claim file after its root and subject are validated."""

    try:
        claim_bytes = read_stable_bytes(claim_path)
        claim = loads_canonical_json(claim_bytes)
        if not isinstance(claim, dict):
            raise RunOwnershipError("run_ownership_claim_conflict")
        version = claim.get("schema_version")
        expected_fields = {
            "schema_version",
            "run_id",
            "claim_identity",
        }
        if version == "v2":
            expected_fields.update({"request_id", "artifact_id", "attempt_number"})
        elif version != "v1":
            raise RunOwnershipError("run_ownership_claim_conflict")
        if set(claim) != expected_fields or claim.get("run_id") != run_id:
            raise RunOwnershipError("run_ownership_claim_conflict")
        raw_identity = claim.get("claim_identity")
        if not isinstance(raw_identity, dict):
            raise RunOwnershipError("run_ownership_claim_conflict")
        claim_identity = parse_identity(raw_identity, field="run ownership claim")
        if claim_bytes != dumps_canonical_json(claim):
            raise RunOwnershipError("run_ownership_claim_conflict")
        if version == "v1":
            return claim_identity, None, None, None
        request_id = claim.get("request_id")
        artifact_id = claim.get("artifact_id")
        attempt_number = claim.get("attempt_number")
        if (
            not isinstance(request_id, str)
            or not isinstance(artifact_id, str)
            or type(attempt_number) is not int
            or attempt_number < 1
        ):
            raise RunOwnershipError("run_ownership_claim_conflict")
        require_identifier("request_id", request_id)
        require_identifier("artifact_id", artifact_id)
        return claim_identity, request_id, artifact_id, attempt_number
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
    *,
    request_id: str | None = None,
    artifact_id: str | None = None,
    attempt_number: int | None = None,
) -> Iterator[RunOwnershipLease]:
    """Hold one non-blocking OS lease bound to a durable immutable claim."""

    with _claim_run_ownership(
        private_root,
        run_id,
        claim_identity,
        allow_create=True,
        require_resolved=True,
        request_id=request_id,
        artifact_id=artifact_id,
        attempt_number=attempt_number,
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
    request_id: str | None = None,
    artifact_id: str | None = None,
    attempt_number: int | None = None,
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
    context_values = (request_id, artifact_id, attempt_number)
    if any(value is not None for value in context_values):
        if (
            not allow_create
            or not isinstance(request_id, str)
            or not isinstance(artifact_id, str)
            or type(attempt_number) is not int
            or attempt_number < 1
        ):
            raise RunOwnershipError("run_ownership_subject_invalid")
        try:
            require_identifier("request_id", request_id)
            require_identifier("artifact_id", artifact_id)
        except RecordValidationError:
            raise RunOwnershipError("run_ownership_subject_invalid") from None
    root = private_root / "runtime-ownership" / run_id
    claim_path = root / "claim.json"
    lock_path = root / "lease.lock"
    resolution_path = root / "resolved.json"
    claim_fields: dict[str, object] = {
        "schema_version": "v1",
        "run_id": run_id,
        "claim_identity": identity_fields(claim_identity),
    }
    expected_context: tuple[str, str, int] | None = None
    if (
        request_id is not None
        and artifact_id is not None
        and attempt_number is not None
    ):
        claim_fields.update(
            {
                "schema_version": "v2",
                "request_id": request_id,
                "artifact_id": artifact_id,
                "attempt_number": attempt_number,
            }
        )
        expected_context = (request_id, artifact_id, attempt_number)
    claim = dumps_canonical_json(claim_fields)
    resolution = dumps_canonical_json(
        {
            "schema_version": "v1",
            "run_id": run_id,
            "claim_identity": identity_fields(claim_identity),
            "resolved": True,
        }
    )
    claim_created = False
    handle: Any | None = None
    locked = False
    try:
        if allow_create:
            with _claim_ownership_namespace(private_root):
                handle, claim_created = _open_locked_run_claim(
                    root,
                    claim_path,
                    lock_path,
                    resolution_path,
                    claim,
                    run_id,
                    claim_identity,
                    expected_context,
                    allow_create=True,
                )
                locked = True
        else:
            handle, claim_created = _open_locked_run_claim(
                root,
                claim_path,
                lock_path,
                resolution_path,
                claim,
                run_id,
                claim_identity,
                expected_context,
                allow_create=False,
            )
            locked = True
    except RunOwnershipError:
        raise
    except (OSError, SafeIoError, UnsafeFilesystemPath):
        raise RunOwnershipError("run_ownership_unavailable") from None
    assert handle is not None
    try:
        try:
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
            pass
    finally:
        if locked:
            try:
                handle.seek(0)
                _unlock(handle)
            except OSError:
                pass
        handle.close()


def _open_locked_run_claim(
    root: Path,
    claim_path: Path,
    lock_path: Path,
    resolution_path: Path,
    claim: bytes,
    run_id: str,
    claim_identity: ContentIdentity,
    expected_context: tuple[str, str, int] | None,
    *,
    allow_create: bool,
) -> tuple[Any, bool]:
    """Open and lock one complete claim while its namespace is coordinated."""

    claim_created = False
    descriptor: int | None = None
    handle: Any | None = None
    locked = False
    try:
        if allow_create:
            ensure_safe_directory(root)
            try:
                write_once_bytes(claim_path, claim)
                claim_created = True
            except FileExistsError:
                existing_claim = _read_run_claim_path(claim_path, run_id)
                existing_context = (
                    None
                    if existing_claim[1] is None
                    else (existing_claim[1], existing_claim[2], existing_claim[3])
                )
                if existing_claim[0] != claim_identity or (
                    existing_context is not None
                    and expected_context is not None
                    and existing_context != expected_context
                ):
                    raise RunOwnershipError("run_ownership_claim_conflict")
        else:
            require_safe_directory(root)
            if _read_run_claim_path(claim_path, run_id)[0] != claim_identity:
                raise RunOwnershipError("run_ownership_claim_conflict")
        existing = safe_path_stat(lock_path, allow_missing=True)
        if existing is None:
            if (
                not allow_create
                or safe_path_stat(resolution_path, allow_missing=True) is not None
            ):
                raise RunOwnershipError("run_ownership_path_invalid")
        elif not stat.S_ISREG(existing.st_mode):
            raise RunOwnershipError("run_ownership_path_invalid")
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if allow_create:
            flags |= os.O_CREAT
        descriptor = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, "r+b")
        descriptor = None
        opened = os.fstat(handle.fileno())
        current = require_safe_regular_file(lock_path)
        if (
            is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or not same_file_object(opened, current)
            or opened.st_nlink != 1
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
            raise RunOwnershipError("run_ownership_path_invalid")
        return handle, claim_created
    except Exception:
        if locked and handle is not None:
            try:
                handle.seek(0)
                _unlock(handle)
            except OSError:
                pass
        if handle is not None:
            handle.close()
        elif descriptor is not None:
            os.close(descriptor)
        raise


@contextmanager
def _claim_ownership_namespace(private_root: Path) -> Iterator[None]:
    """Serialize publication and discovery of complete run-claim directories."""

    ownership_root = private_root / "runtime-ownership"
    lock_path = ownership_root / ".claims.lock"
    descriptor: int | None = None
    try:
        ensure_safe_directory(ownership_root)
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
    except (OSError, UnsafeFilesystemPath):
        raise RunOwnershipError("run_ownership_unavailable") from None
    assert descriptor is not None
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
                raise RunOwnershipError("run_ownership_path_invalid")
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            elif size != 1:
                raise RunOwnershipError("run_ownership_path_invalid")
            handle.seek(0)
            _lock_blocking(handle)
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
                raise RunOwnershipError("run_ownership_path_invalid")
            yield
        except RunOwnershipError:
            raise
        except (OSError, SafeIoError, UnsafeFilesystemPath):
            raise RunOwnershipError("run_ownership_unavailable") from None
        finally:
            if locked:
                try:
                    handle.seek(0)
                    _unlock(handle)
                except OSError:
                    pass


def _validate_unpublished_claim_directory(root: Path) -> None:
    """Accept only an empty or write-once temporary directory from a lost creator."""

    with os.scandir(root) as iterator:
        entries = tuple(iterator)
    for entry in entries:
        info = entry.stat(follow_symlinks=False)
        prefix = ".claim.json."
        suffix = ".tmp"
        middle = entry.name[len(prefix) : -len(suffix)]
        if (
            is_link_or_reparse(info)
            or not stat.S_ISREG(info.st_mode)
            or not entry.name.startswith(prefix)
            or not entry.name.endswith(suffix)
            or len(middle) != 32
            or any(character not in "0123456789abcdef" for character in middle)
        ):
            raise RunOwnershipError("run_ownership_path_invalid")


def _repair_unpublished_run_lock(root: Path) -> None:
    """Finish only the claim-before-lock crash window under the namespace lease."""

    lock_path = root / "lease.lock"
    resolution_path = root / "resolved.json"
    existing = safe_path_stat(lock_path, allow_missing=True)
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise RunOwnershipError("run_ownership_path_invalid")
    if existing is not None and existing.st_size == 1:
        return
    if safe_path_stat(resolution_path, allow_missing=True) is not None:
        raise RunOwnershipError("run_ownership_path_invalid")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        opened = os.fstat(handle.fileno())
        current = require_safe_regular_file(lock_path)
        if (
            is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or not same_file_object(opened, current)
            or opened.st_nlink != 1
        ):
            raise RunOwnershipError("run_ownership_path_invalid")
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        elif size != 1:
            raise RunOwnershipError("run_ownership_path_invalid")
        handle.seek(0)
        if handle.read(1) != b"\0":
            raise RunOwnershipError("run_ownership_path_invalid")


def _lock_blocking(handle: Any) -> None:
    try:
        if os.name == "nt":
            msvcrt: Any = importlib.import_module("msvcrt")
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError:
        raise RunOwnershipError("run_ownership_unavailable") from None


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
