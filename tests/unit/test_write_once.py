import pytest

from temper_ml.domain.projections import HashProjection
from temper_ml.store.write_once import WriteOnceExists, WriteOnceStore


def test_write_once_store_creates_immutable_identity_and_refuses_overwrite(tmp_path):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    record = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }

    written = store.write_projected_json("datasets", projection, record)

    assert written.path.exists()
    assert written.path.read_bytes() == (
        b'{"dataset_id":"dataset-synthetic-demo","record_count":2,"schema_version":1}\n'
    )

    with pytest.raises(WriteOnceExists):
        store.write_projected_json("datasets", projection, record)


def test_write_once_store_keeps_immutable_evidence_separate_from_derived_state(tmp_path):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="artifact", version="v1")
    evidence = {"artifact_id": "artifact-synthetic-demo", "schema_version": 1}

    written = store.write_projected_json("artifacts", projection, evidence)
    store.write_derived_state("registry/state", {"latest": "mutable-view"})

    assert "immutable" in written.path.parts
    assert (tmp_path / "derived" / "registry" / "state.json").exists()
    assert store.read_projected_json("artifacts", written.identity) == evidence
