"""Portable logical locations for a Windows host and WSL worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
import re


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class PortablePathError(ValueError):
    """A stable portable-path validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PortableLocation:
    """A canonical-manifest-safe, project-relative logical location."""

    logical_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.logical_path, str) or "\\" in self.logical_path:
            raise PortablePathError("portable_location_invalid")
        path = PurePosixPath(self.logical_path)
        if (
            not self.logical_path
            or _WINDOWS_DRIVE.match(self.logical_path)
            or "://" in self.logical_path
            or path.is_absolute()
            or path == PurePosixPath(".")
            or ".." in path.parts
            or str(path) != self.logical_path
        ):
            raise PortablePathError("portable_location_invalid")

    def to_dict(self) -> dict[str, str]:
        return {"logical_path": self.logical_path}


@dataclass(frozen=True)
class WindowsWslPathMap:
    """Map runtime paths while keeping roots out of canonical locations."""

    host_root: PureWindowsPath
    worker_root: PurePosixPath

    def __post_init__(self) -> None:
        if (
            not isinstance(self.host_root, PureWindowsPath)
            or not self.host_root.is_absolute()
        ):
            raise PortablePathError("host_root_invalid")
        if (
            not isinstance(self.worker_root, PurePosixPath)
            or not self.worker_root.is_absolute()
        ):
            raise PortablePathError("worker_root_invalid")

    def portable_from_host(self, path: PureWindowsPath) -> PortableLocation:
        if not isinstance(path, PureWindowsPath) or not path.is_absolute():
            raise PortablePathError("host_path_invalid")
        try:
            relative = path.relative_to(self.host_root)
        except ValueError:
            raise PortablePathError("host_path_outside_project") from None
        if relative == PureWindowsPath("."):
            raise PortablePathError("portable_location_invalid")
        return PortableLocation(PurePosixPath(*relative.parts).as_posix())

    def host_path(self, location: PortableLocation) -> PureWindowsPath:
        if not isinstance(location, PortableLocation):
            raise PortablePathError("portable_location_invalid")
        return self.host_root.joinpath(*PurePosixPath(location.logical_path).parts)

    def worker_path(self, location: PortableLocation) -> PurePosixPath:
        if not isinstance(location, PortableLocation):
            raise PortablePathError("portable_location_invalid")
        return self.worker_root.joinpath(location.logical_path)
