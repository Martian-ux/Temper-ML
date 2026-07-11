from pathlib import Path

import pytest

from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json
from temper_ml.store.event_stream import (
    EventRequest,
    EventStream,
    EventStreamCorrupt,
)


def _events(stream: EventStream) -> list[Path]:
    return sorted(stream.directory.glob("*.json"))


def test_incomplete_temporary_files_are_ignored(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stored = stream.append(EventRequest("one", "created", {"ok": True}))
    (stream.directory / ".event.interrupted.tmp").write_bytes(b'{"partial":')

    assert stream.read_verified() == (stored,)


def test_unrecognized_non_temporary_entry_fails_closed(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stream.append(EventRequest("one", "created", {}))
    (stream.directory / "unrecognized.partial").write_bytes(
        b"not a Temper temporary file"
    )

    with pytest.raises(EventStreamCorrupt, match="unexpected"):
        stream.read_verified()


def test_sequence_gap_fails_closed(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stream.append(EventRequest("one", "created", {}))
    second = stream.append(EventRequest("two", "updated", {}))
    second.path.rename(
        second.path.with_name(
            second.path.name.replace("00000000000000000002", "00000000000000000003")
        )
    )

    with pytest.raises(EventStreamCorrupt, match="sequence"):
        stream.read_verified()


def test_wrong_predecessor_fails_closed(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stream.append(EventRequest("one", "created", {}))
    stream.append(EventRequest("two", "updated", {}))
    path = _events(stream)[1]
    envelope = loads_canonical_json(path.read_bytes())
    envelope["predecessor_hash"] = "0" * 64
    path.write_bytes(dumps_canonical_json(envelope))

    with pytest.raises(EventStreamCorrupt, match="predecessor"):
        stream.read_verified()


def test_filename_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stored = stream.append(EventRequest("one", "created", {}))
    stored.path.rename(stored.path.with_name(f"{1:020d}-{'0' * 64}.json"))

    with pytest.raises(EventStreamCorrupt, match="filename"):
        stream.read_verified()


def test_tampered_event_content_blocks_rebuild(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stored = stream.append(EventRequest("one", "counter.added", {"amount": 1}))
    envelope = loads_canonical_json(stored.path.read_bytes())
    envelope["payload"]["amount"] = 100
    stored.path.write_bytes(dumps_canonical_json(envelope))

    with pytest.raises(EventStreamCorrupt):
        stream.rebuild(0, lambda state, event: state + event.payload["amount"])


def test_duplicate_idempotency_keys_in_chain_fail_closed(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stream.append(EventRequest("one", "created", {}))
    second = stream.append(EventRequest("two", "updated", {}))
    envelope = loads_canonical_json(second.path.read_bytes())
    envelope["idempotency_key"] = "one"
    second.path.write_bytes(dumps_canonical_json(envelope))

    with pytest.raises(EventStreamCorrupt, match="idempotency"):
        stream.read_verified()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "v2"),
        ("projection_version", "v2"),
        ("sequence", True),
        ("event_type", ""),
        ("payload", []),
    ],
)
def test_invalid_envelope_fields_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    stream = EventStream(tmp_path / "events")
    stored = stream.append(EventRequest("one", "created", {}))
    envelope = loads_canonical_json(stored.path.read_bytes())
    envelope[field] = value
    stored.path.write_bytes(dumps_canonical_json(envelope))

    with pytest.raises(EventStreamCorrupt):
        stream.read_verified()
