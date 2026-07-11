"""Versioned content identity projections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping

from temper_ml.store.canonical_json import dumps_canonical_json

_PROJECTION_NAME = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_PROJECTION_VERSION = re.compile(r"^v[1-9][0-9]*$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class ProjectionError(ValueError):
    """Raised when an identity projection is malformed."""


@dataclass(frozen=True)
class HashProjection:
    """A named, versioned Temper hash projection."""

    name: str
    version: str

    def __post_init__(self) -> None:
        if not _PROJECTION_NAME.fullmatch(self.name):
            raise ProjectionError(f"invalid projection name: {self.name!r}")
        if not _PROJECTION_VERSION.fullmatch(self.version):
            raise ProjectionError(f"invalid projection version: {self.version!r}")

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class ContentIdentity:
    """A content identity with an explicit digest algorithm."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ProjectionError(
                f"unsupported content identity algorithm: {self.algorithm!r}"
            )
        if not _SHA256_HEX.fullmatch(self.value):
            raise ProjectionError(f"invalid sha256 identity value: {self.value!r}")

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.value}"


def projection_preimage(
    projection: HashProjection, projected_fields: Mapping[str, Any]
) -> bytes:
    """Return the exact bytes hashed for a Temper-owned projection."""

    return f"temper:{projection.label}\n".encode("utf-8") + dumps_canonical_json(
        projected_fields
    )


def content_identity(
    projection: HashProjection, projected_fields: Mapping[str, Any]
) -> ContentIdentity:
    """Hash projected fields with a Temper domain prefix and projection version."""

    digest = hashlib.sha256(
        projection_preimage(projection, projected_fields)
    ).hexdigest()
    return ContentIdentity(algorithm="sha256", value=digest)
