"""Fail-closed verification for byte and bundle identities."""

from __future__ import annotations

from pathlib import Path

from temper_ml.domain.artifacts import (
    ArtifactError,
    BUNDLE_PROJECTION,
    BundleManifest,
    build_bundle_manifest,
    file_identity,
)
from temper_ml.domain.projections import ContentIdentity, content_identity


class VerificationError(RuntimeError):
    """Raised when stored bytes do not match their claimed identity."""


def verify_file(path: Path | str, expected: ContentIdentity) -> None:
    """Verify a regular file's byte identity."""

    try:
        actual = file_identity(path)
    except (ArtifactError, OSError) as exc:
        raise VerificationError(f"unable to verify artifact file: {path!s}") from exc
    if actual != expected:
        raise VerificationError(
            f"artifact identity mismatch: expected {expected}, found {actual}"
        )


def verify_bundle(root: Path | str, manifest: BundleManifest) -> None:
    """Rebuild and compare a bundle manifest, including its claimed identity."""

    projected_identity = content_identity(BUNDLE_PROJECTION, manifest.projected_fields())
    if projected_identity != manifest.identity:
        raise VerificationError("bundle manifest identity mismatch")
    try:
        actual = build_bundle_manifest(root, [member.path for member in manifest.members])
    except (ArtifactError, OSError) as exc:
        raise VerificationError("bundle member verification failed") from exc
    if actual != manifest:
        raise VerificationError("bundle member identity or size mismatch")
