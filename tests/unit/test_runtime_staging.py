from pathlib import Path

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.paths import PortableLocation
from temper_ml.runtime.staging import (
    StagingError,
    TransferDirection,
    TransferManifest,
    TransferReceipt,
    build_transfer_manifest,
    read_verified_transfer,
    stage_transfer,
    verify_transfer,
)


def test_transfer_round_trip_is_content_identified_and_portable(
    tmp_path: Path,
) -> None:
    first = PortableLocation("inputs/dataset.jsonl")
    second = PortableLocation("inputs/checkpoint.bin")
    payloads = {first: b"dataset\n", second: b"checkpoint"}
    manifest = build_transfer_manifest(
        TransferDirection.HOST_TO_WORKER,
        {
            first: ("rendered_dataset", payloads[first]),
            second: ("resume_checkpoint", payloads[second]),
        },
    )
    receipt = stage_transfer(tmp_path, manifest, payloads)

    assert TransferManifest.from_dict(manifest.to_dict()) == manifest
    assert TransferReceipt.from_dict(receipt.to_dict()) == receipt
    assert verify_transfer(tmp_path, manifest) == receipt
    assert read_verified_transfer(tmp_path, manifest) == payloads
    assert all(
        not member.logical_location.logical_path.startswith(("/", "C:"))
        for member in receipt.verified_members
    )


def test_partial_replaced_or_corrupt_transfer_never_verifies(tmp_path: Path) -> None:
    location = PortableLocation("outputs/adapter.bin")
    manifest = build_transfer_manifest(
        TransferDirection.WORKER_TO_HOST,
        {location: ("adapter_payload", b"verified")},
    )
    with pytest.raises(StagingError, match="transfer_member_unavailable"):
        verify_transfer(tmp_path, manifest)

    stage_transfer(tmp_path, manifest, {location: b"verified"})
    (tmp_path / "outputs" / "adapter.bin").write_bytes(b"replaced")
    with pytest.raises(StagingError, match="transfer_member_identity_mismatch"):
        verify_transfer(tmp_path, manifest)
    with pytest.raises(StagingError, match="transfer_existing_member_conflict"):
        stage_transfer(tmp_path, manifest, {location: b"verified"})


def test_manifest_and_receipt_identity_tampering_is_rejected(tmp_path: Path) -> None:
    del tmp_path
    location = PortableLocation("outputs/result.json")
    manifest = build_transfer_manifest(
        TransferDirection.WORKER_TO_HOST,
        {location: ("inference_result", b"{}")},
    )
    value = manifest.to_dict()
    value["identity"] = {"algorithm": "sha256", "value": "0" * 64}
    with pytest.raises(StagingError, match="transfer_manifest_identity_mismatch"):
        TransferManifest.from_dict(value)

    with pytest.raises(StagingError, match="transfer_receipt_manifest_mismatch"):
        TransferReceipt(
            manifest.direction,
            ContentIdentity("sha256", "1" * 64),
            manifest.members,
            True,
        )
