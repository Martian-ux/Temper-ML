from pathlib import Path

import pytest

from temper_ml.domain.artifacts import (
    ArtifactError,
    build_bundle_manifest,
    byte_identity,
    file_identity,
)
from temper_ml.store.verifier import VerificationError, verify_bundle, verify_file


def test_byte_and_streamed_file_identity_depend_only_on_bytes(tmp_path: Path) -> None:
    payload = b"tiny synthetic artifact\n"
    first = tmp_path / "first.bin"
    second = tmp_path / "elsewhere" / "renamed.bin"
    first.write_bytes(payload)
    second.parent.mkdir()
    second.write_bytes(payload)

    assert byte_identity(payload) == file_identity(first, chunk_size=3)
    assert file_identity(first) == file_identity(second)


def test_file_verification_fails_closed_after_tampering(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"synthetic-v1")
    expected = file_identity(path)
    verify_file(path, expected)

    path.write_bytes(b"synthetic-v2")
    with pytest.raises(VerificationError, match="identity mismatch"):
        verify_file(path, expected)


def test_bundle_identity_is_stable_across_roots_and_input_order(tmp_path: Path) -> None:
    roots = [tmp_path / "one", tmp_path / "other"]
    for root in roots:
        (root / "nested").mkdir(parents=True)
        (root / "alpha.txt").write_bytes(b"alpha")
        (root / "nested" / "beta.txt").write_bytes(b"beta")

    first = build_bundle_manifest(roots[0], ["nested/beta.txt", "alpha.txt"])
    second = build_bundle_manifest(roots[1], ["alpha.txt", "nested/beta.txt"])

    assert first == second
    assert first.identity == second.identity
    verify_bundle(roots[0], first)


def test_bundle_verification_detects_tampered_member(tmp_path: Path) -> None:
    (tmp_path / "member.txt").write_bytes(b"before")
    manifest = build_bundle_manifest(tmp_path)
    (tmp_path / "member.txt").write_bytes(b"after")

    with pytest.raises(VerificationError, match="member"):
        verify_bundle(tmp_path, manifest)


def test_bundle_rejects_non_regular_member(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()
    with pytest.raises(ArtifactError, match="regular file"):
        build_bundle_manifest(tmp_path, ["directory"])
