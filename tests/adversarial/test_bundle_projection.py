import os
from pathlib import Path

import pytest

from temper_ml.domain.artifacts import (
    ArtifactError,
    BundleManifest,
    build_bundle_manifest,
)
from temper_ml.store.verifier import VerificationError, verify_bundle


@pytest.mark.parametrize(
    "member",
    [
        "/absolute.bin",
        "\\absolute.bin",
        "C:/absolute.bin",
        "../escape.bin",
        "nested/../escape.bin",
        "./member.bin",
        "nested\\member.bin",
    ],
)
def test_bundle_rejects_ambiguous_or_escaping_member_paths(
    tmp_path: Path, member: str
) -> None:
    with pytest.raises(ArtifactError):
        build_bundle_manifest(tmp_path, [member])


def test_bundle_rejects_duplicate_normalized_members(tmp_path: Path) -> None:
    (tmp_path / "member.bin").write_bytes(b"public synthetic bytes")
    with pytest.raises(ArtifactError, match="duplicate"):
        build_bundle_manifest(tmp_path, ["member.bin", "member.bin"])


def test_bundle_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    link = tmp_path / "link.bin"
    target.write_bytes(b"public synthetic bytes")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")

    with pytest.raises(ArtifactError, match="symlink"):
        build_bundle_manifest(tmp_path)


def test_bundle_verification_rejects_tampered_manifest(tmp_path: Path) -> None:
    (tmp_path / "member.bin").write_bytes(b"public synthetic bytes")
    manifest = build_bundle_manifest(tmp_path)
    member = manifest.members[0]
    tampered = BundleManifest(
        schema_version=manifest.schema_version,
        projection_version=manifest.projection_version,
        members=(type(member)(member.path, member.identity, member.size + 1),),
        identity=manifest.identity,
    )

    with pytest.raises(VerificationError):
        verify_bundle(tmp_path, tampered)


def test_bundle_rejects_fifo_when_supported(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    fifo = tmp_path / "member.pipe"
    os.mkfifo(fifo)
    with pytest.raises(ArtifactError, match="regular file"):
        build_bundle_manifest(tmp_path)
