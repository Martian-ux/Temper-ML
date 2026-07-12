"""Fail-closed helpers for local paths that must not traverse links."""

from __future__ import annotations

import os
from pathlib import Path
import stat


class UnsafeFilesystemPath(RuntimeError):
    """Raised when a path is missing, linked, or has an unexpected type."""


def is_link_or_reparse(info: os.stat_result) -> bool:
    """Return whether an lstat/fstat result represents a link or reparse point."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def safe_path_stat(
    path: Path | str, *, allow_missing: bool = False
) -> os.stat_result | None:
    """lstat every component without resolving links and return the leaf stat."""

    absolute = _absolute_without_resolving(path)
    leaf: os.stat_result | None = None
    for current in _components(absolute):
        try:
            leaf = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                return None
            raise
        if is_link_or_reparse(leaf):
            raise UnsafeFilesystemPath(
                f"symlinks and reparse points are not allowed: {current}"
            )
    return leaf


def ensure_safe_directory(path: Path | str) -> Path:
    """Create a directory one component at a time without traversing links."""

    absolute = _absolute_without_resolving(path)
    for current in _components(absolute):
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            info = current.lstat()
        if is_link_or_reparse(info):
            raise UnsafeFilesystemPath(
                f"symlinks and reparse points are not allowed: {current}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise UnsafeFilesystemPath(f"path component is not a directory: {current}")
    return absolute


def require_safe_directory(path: Path | str) -> os.stat_result:
    """Require an existing non-linked directory and safe ancestors."""

    info = safe_path_stat(path)
    assert info is not None
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeFilesystemPath(f"path is not a directory: {path}")
    return info


def require_safe_regular_file(path: Path | str) -> os.stat_result:
    """Require an existing non-linked regular file and safe ancestors."""

    info = safe_path_stat(path)
    assert info is not None
    if not stat.S_ISREG(info.st_mode):
        raise UnsafeFilesystemPath(f"path is not a regular file: {path}")
    return info


def same_file_object(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare stable filesystem identity fields for two observations."""

    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _absolute_without_resolving(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _components(path: Path) -> tuple[Path, ...]:
    parts = path.parts
    if not parts:
        raise UnsafeFilesystemPath("filesystem path must not be empty")
    current = Path(parts[0])
    values = [current]
    for part in parts[1:]:
        current = current / part
        values.append(current)
    return tuple(values)
