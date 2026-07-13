from dataclasses import replace
from pathlib import Path

import pytest

from temper_ml.domain.artifacts import build_bundle_manifest
from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactIntegrityExpectation,
    verify_artifact_bundle,
)
from temper_ml.runtime.fixture_adapter import FixtureAdapter
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json

from test_fixture_adapter import _adapter_components


def _materialize(root: Path, members) -> None:
    root.mkdir(parents=True)
    for name, data in members.items():
        (root / name).write_bytes(data)


def _fixture(tmp_path: Path):
    request, model, group = _adapter_components(tmp_path / "records")
    output = FixtureAdapter().execute(request)
    root = tmp_path / "artifact"
    _materialize(root, output.members)
    assert output.bundle_manifest is not None
    expectation = ArtifactIntegrityExpectation(
        output.bundle_manifest.identity,
        request.run,
        request.runtime_request,
        request.experiment,
        request.recipe_resolution,
        request.dataset_version,
        model,
        group,
    )
    return root, output, expectation


def test_integrity_verifies_structure_bytes_provenance_and_compatibility(
    tmp_path: Path,
) -> None:
    root, output, expectation = _fixture(tmp_path)

    result = verify_artifact_bundle(root, expectation)

    assert result.bundle_manifest == output.bundle_manifest
    assert result.to_receipt()["structure_verified"] is True
    assert result.to_receipt()["provenance_verified"] is True
    assert result.to_receipt()["compatibility_verified"] is True


def test_integrity_rejects_partial_or_transferred_mismatched_bytes(
    tmp_path: Path,
) -> None:
    root, _, expectation = _fixture(tmp_path)
    (root / "provenance.json").unlink()

    with pytest.raises(ArtifactIntegrityError, match="artifact_structure_mismatch"):
        verify_artifact_bundle(root, expectation)

    root, _, expectation = _fixture(tmp_path / "mismatch")
    (root / "adapter.bin").write_bytes(b"partial")
    with pytest.raises(
        ArtifactIntegrityError, match="artifact_content_identity_mismatch"
    ):
        verify_artifact_bundle(root, expectation)


def test_integrity_rejects_well_formed_counterfeit_final_state(
    tmp_path: Path,
) -> None:
    root, _, expectation = _fixture(tmp_path)
    provenance_path = root / "provenance.json"
    provenance = loads_canonical_json(provenance_path.read_bytes())
    assert isinstance(provenance, dict)
    provenance["final_training_state_identity"] = {
        "algorithm": "sha256",
        "value": "f" * 64,
    }
    provenance_path.write_bytes(dumps_canonical_json(provenance))
    tampered_manifest = build_bundle_manifest(root)

    with pytest.raises(ArtifactIntegrityError, match="artifact_provenance_mismatch"):
        verify_artifact_bundle(
            root,
            replace(expectation, bundle_identity=tampered_manifest.identity),
        )


def test_integrity_expectation_rejects_incompatible_base_model(
    tmp_path: Path,
) -> None:
    _, _, expectation = _fixture(tmp_path)
    incompatible = replace(
        expectation.base_model_revision,
        model_id="model-incompatible",
        tokenizer_identity=ContentIdentity("sha256", "e" * 64),
    )

    with pytest.raises(ArtifactIntegrityError, match="artifact_base_model_mismatch"):
        replace(expectation, base_model_revision=incompatible)
