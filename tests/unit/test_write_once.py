from pathlib import Path

import pytest

from temper_ml.domain.projections import HashProjection
from temper_ml.store.canonical_json import dumps_canonical_json
import temper_ml.store.write_once as write_once
from temper_ml.store.write_once import (
    WriteOnceCorrupt,
    WriteOnceError,
    WriteOnceExists,
    WriteOnceStore,
)


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


def test_write_once_store_keeps_immutable_evidence_separate_from_derived_state(
    tmp_path,
):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="artifact", version="v1")
    evidence = {"artifact_id": "artifact-synthetic-demo", "schema_version": 1}

    written = store.write_projected_json("artifacts", projection, evidence)
    store.write_derived_state("registry/state", {"latest": "mutable-view"})

    assert "immutable" in written.path.parts
    assert (tmp_path / "derived" / "registry" / "state.json").exists()
    assert (
        store.read_projected_json("artifacts", projection, written.identity) == evidence
    )


def test_read_projected_json_rejects_tampered_immutable_content(tmp_path):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    evidence = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }
    tampered = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 3,
        "schema_version": 1,
    }
    written = store.write_projected_json("datasets", projection, evidence)
    written.path.write_bytes(dumps_canonical_json(tampered))

    with pytest.raises(WriteOnceCorrupt):
        store.read_projected_json("datasets", projection, written.identity)


def test_read_projected_json_rejects_noncanonical_immutable_content(tmp_path):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    evidence = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }
    written = store.write_projected_json("datasets", projection, evidence)
    written.path.write_bytes(
        b'{ "schema_version" : 1, "record_count" : 2, '
        b'"dataset_id" : "dataset-synthetic-demo" }\n'
    )

    with pytest.raises(WriteOnceCorrupt, match="canonical JSON"):
        store.read_projected_json("datasets", projection, written.identity)


def test_read_projected_json_normalizes_invalid_utf8_as_corruption(tmp_path):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    evidence = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }
    written = store.write_projected_json("datasets", projection, evidence)
    written.path.write_bytes(b"\xff")

    with pytest.raises(WriteOnceCorrupt, match="canonical JSON"):
        store.read_projected_json("datasets", projection, written.identity)


def test_write_projected_json_normalizes_concurrent_create_as_existing(
    tmp_path, monkeypatch
):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    evidence = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }

    def concurrent_write(path, payload):
        Path(path).write_bytes(dumps_canonical_json(evidence))
        raise FileExistsError

    monkeypatch.setattr(write_once, "write_once_bytes", concurrent_write)

    with pytest.raises(WriteOnceExists):
        store.write_projected_json("datasets", projection, evidence)


def test_incomplete_temporary_record_does_not_block_atomic_retry(tmp_path):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    evidence = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }
    identity = write_once.content_identity(projection, evidence)
    final_path = store._immutable_path("datasets", identity)
    final_path.parent.mkdir(parents=True)
    interrupted = final_path.with_name(f".{final_path.name}.interrupted.tmp")
    interrupted.write_bytes(b'{"partial":')

    written = store.write_projected_json("datasets", projection, evidence)

    assert written.path == final_path
    assert store.read_projected_json("datasets", projection, identity) == evidence
    assert interrupted.read_bytes() == b'{"partial":'


def test_write_projected_json_rejects_symlinked_store_directory(tmp_path):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    target = tmp_path / "outside"
    target.mkdir()
    immutable_directory = tmp_path / "immutable"
    immutable_directory.mkdir()
    dataset_directory = immutable_directory / "datasets"
    try:
        dataset_directory.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(WriteOnceError, match="symlink|reparse"):
        store.write_projected_json(
            "datasets",
            projection,
            {
                "dataset_id": "dataset-synthetic-demo",
                "record_count": 2,
                "schema_version": 1,
            },
        )


def test_read_projected_json_rejects_wrong_projection_version(tmp_path):
    store = WriteOnceStore(tmp_path)
    write_projection = HashProjection(name="dataset_version", version="v1")
    read_projection = HashProjection(name="dataset_version", version="v2")
    evidence = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }
    written = store.write_projected_json("datasets", write_projection, evidence)

    with pytest.raises(WriteOnceCorrupt):
        store.read_projected_json("datasets", read_projection, written.identity)


@pytest.mark.parametrize(
    "unsafe_area", ["/datasets", "../datasets", "datasets/../other"]
)
def test_write_once_store_rejects_unsafe_immutable_area_paths(tmp_path, unsafe_area):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")
    evidence = {
        "dataset_id": "dataset-synthetic-demo",
        "record_count": 2,
        "schema_version": 1,
    }
    written = store.write_projected_json("datasets", projection, evidence)

    with pytest.raises(WriteOnceError):
        store.write_projected_json(unsafe_area, projection, evidence)

    with pytest.raises(WriteOnceError):
        store.read_projected_json(unsafe_area, projection, written.identity)


@pytest.mark.parametrize(
    "area", ["nul", "nested/con", "trailing.", "trailing ", "bad\x00"]
)
def test_write_once_store_rejects_non_portable_area_paths(tmp_path, area):
    store = WriteOnceStore(tmp_path)
    projection = HashProjection(name="dataset_version", version="v1")

    with pytest.raises(WriteOnceError, match="non-portable"):
        store.write_projected_json(area, projection, {"value": 1})
