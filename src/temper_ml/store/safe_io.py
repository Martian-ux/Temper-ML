"""Stable, link-safe byte I/O for canonical local evidence."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from uuid import uuid4

from temper_ml.filesystem import (
    UnsafeFilesystemPath,
    ensure_safe_directory,
    is_link_or_reparse,
    require_safe_regular_file,
    safe_path_stat,
    same_file_object,
)


class SafeIoError(RuntimeError):
    """Raised when canonical byte I/O cannot prove stable safe paths."""


def read_stable_bytes(path: Path | str) -> bytes:
    """Read one regular file while proving the opened object stayed stable."""

    candidate = Path(path)
    try:
        before = require_safe_regular_file(candidate)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                is_link_or_reparse(opened)
                or not stat.S_ISREG(opened.st_mode)
                or not same_file_object(before, opened)
            ):
                raise SafeIoError("canonical file changed while opening")
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = require_safe_regular_file(candidate)
    except SafeIoError:
        raise
    except (OSError, UnsafeFilesystemPath) as exc:
        raise SafeIoError("unable to read a stable canonical file") from exc
    if (
        not same_file_object(opened, after)
        or not same_file_object(after, current)
        or len(payload) != after.st_size
    ):
        raise SafeIoError("canonical file changed while reading")
    return payload


def write_once_bytes(path: Path | str, payload: bytes) -> None:
    """Atomically commit bytes without ever replacing an existing final path."""

    candidate = Path(path)
    temp_path = candidate.with_name(f".{candidate.name}.{uuid4().hex}.tmp")
    try:
        ensure_safe_directory(candidate.parent)
        if safe_path_stat(candidate, allow_missing=True) is not None:
            raise FileExistsError(candidate.name)
        if safe_path_stat(temp_path, allow_missing=True) is not None:
            raise SafeIoError("temporary canonical path already exists")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temp_path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            opened = os.fstat(handle.fileno())
            if is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
                raise SafeIoError("temporary canonical path is unsafe")
        current_temp = require_safe_regular_file(temp_path)
        if not same_file_object(opened, current_temp):
            raise SafeIoError("temporary canonical path changed before commit")
        if safe_path_stat(candidate, allow_missing=True) is not None:
            raise FileExistsError(candidate.name)
        try:
            os.link(temp_path, candidate, follow_symlinks=False)
        except TypeError:
            os.link(temp_path, candidate)
        final = require_safe_regular_file(candidate)
        if not same_file_object(current_temp, final):
            raise SafeIoError("canonical path changed during commit")
        _fsync_directory(candidate.parent)
    except (FileExistsError, SafeIoError):
        raise
    except (OSError, UnsafeFilesystemPath) as exc:
        raise SafeIoError("unable to commit canonical bytes safely") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def replace_bytes(path: Path | str, payload: bytes) -> None:
    """Atomically replace rebuildable derived bytes through a safe temp file."""

    candidate = Path(path)
    temp_path = candidate.with_name(f".{candidate.name}.{uuid4().hex}.tmp")
    try:
        ensure_safe_directory(candidate.parent)
        existing = safe_path_stat(candidate, allow_missing=True)
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise SafeIoError("derived state path is not a regular file")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temp_path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            opened = os.fstat(handle.fileno())
        current_temp = require_safe_regular_file(temp_path)
        if (
            is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or not same_file_object(opened, current_temp)
        ):
            raise SafeIoError("temporary derived-state path is unsafe")
        os.replace(temp_path, candidate)
        require_safe_regular_file(candidate)
        _fsync_directory(candidate.parent)
    except SafeIoError:
        raise
    except (OSError, UnsafeFilesystemPath) as exc:
        raise SafeIoError("unable to replace derived state safely") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
