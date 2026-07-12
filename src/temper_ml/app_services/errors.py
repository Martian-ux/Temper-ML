"""Stable, public-safe application-service failures."""

from __future__ import annotations


class ApplicationServiceError(RuntimeError):
    """A bounded service failure represented only by a stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
