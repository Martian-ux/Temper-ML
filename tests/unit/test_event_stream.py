from pathlib import Path

import pytest

from temper_ml.store.event_stream import (
    EventConflict,
    EventRequest,
    EventStream,
)


def test_append_builds_hash_linked_chain_and_is_idempotent(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    first_request = EventRequest("request-1", "run.created", {"value": 1})
    first = stream.append(first_request)
    repeated = stream.append(first_request)
    second = stream.append(EventRequest("request-2", "run.updated", {"value": 2}))

    assert repeated == first
    assert first.sequence == 1
    assert first.predecessor_hash is None
    assert second.sequence == 2
    assert second.predecessor_hash == first.identity.value
    assert first.path.name == f"{1:020d}-{first.identity.value}.json"
    assert [event.sequence for event in stream.read_verified()] == [1, 2]


def test_conflicting_idempotency_key_fails_closed(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stream.append(EventRequest("same-key", "run.created", {"value": 1}))

    with pytest.raises(EventConflict, match="idempotency"):
        stream.append(EventRequest("same-key", "run.created", {"value": 2}))


def test_rebuild_reduces_only_verified_events(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "events")
    stream.append(EventRequest("one", "counter.added", {"amount": 2}))
    stream.append(EventRequest("two", "counter.added", {"amount": 3}))

    result = stream.rebuild(
        0, lambda state, event: state + event.payload["amount"]
    )

    assert result == 5
