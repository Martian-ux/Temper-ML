import hashlib
import os
from pathlib import Path

import pytest

from temper_ml.domain.artifacts import build_bundle_manifest, byte_identity
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.projects import Project
from temper_ml.domain.records import RecordReference, record_reference
from temper_ml.domain.tasks import TaskDefinition
from temper_ml.store.canonical_json import dumps_canonical_json, loads_canonical_json
from temper_ml.store.evidence import (
    EvidenceAmbiguous,
    EvidenceCorrupt,
    EvidenceError,
    EvidenceExists,
    TypedEvidenceStore,
)
from temper_ml.store.event_stream import EventRequest
from temper_ml.store.event_stream import EventStream
from temper_ml.store.redaction import RedactionContext


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _task(
    *,
    task_id: str = "task-synthetic",
    description: str = "Transform synthetic text deterministically.",
    input_schema: dict[str, object] | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        display_name="Synthetic transformation",
        description=description,
        input_schema=input_schema or {"required": ["input"]},
        output_schema={"required": ["output"]},
        rendering_contract=_identity("synthetic-renderer"),
        objectives=("determinism",),
        capabilities=("text_generation",),
    )


def _project(task: TaskDefinition) -> Project:
    return Project(
        project_id="project-synthetic",
        display_name="Synthetic project",
        purpose="Exercise canonical evidence without private inputs.",
        task_definition=record_reference(task),
    )


def _store(root: Path) -> TypedEvidenceStore:
    return TypedEvidenceStore(root, redaction_context=RedactionContext())


def test_typed_records_use_envelope_identity_and_verify_reference_closure(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    task = _task()
    project = _project(task)

    stored_task = store.write_record(task)
    stored_project = store.write_record(project)

    assert stored_task.path.name == f"{task.identity.value}.json"
    assert stored_project.path.name == f"{project.identity.value}.json"
    assert loads_canonical_json(stored_task.path.read_bytes()) == task.to_dict()
    assert store.read_record(record_reference(task)).record == task
    assert store.read_record(record_reference(project)).record == project
    assert store.verify().to_dict() == {
        "status": "verified",
        "record_count": 2,
        "record_counts": {"project": 1, "task_definition": 1},
        "event_stream_count": 0,
        "event_count": 0,
        "bundle_manifest_count": 0,
        "derived_state_rebuildable": True,
    }

    with pytest.raises(EvidenceExists, match="^record_exists$"):
        store.write_record(task)
    with pytest.raises(EvidenceError, match="^immutable_cleanup_forbidden$"):
        store.remove_record(record_reference(task))
    assert stored_task.path.read_bytes() == dumps_canonical_json(task.to_dict())


def test_verification_rejects_dangling_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    task = _task()
    store.write_record(_project(task))

    with pytest.raises(EvidenceCorrupt, match="^dangling_record_reference$"):
        store.verify()


@pytest.mark.parametrize("surface", ["registry", "runs", "artifacts"])
def test_verification_fails_closed_on_unverified_legacy_surfaces(
    tmp_path: Path, surface: str
) -> None:
    temper = tmp_path / ".temper"
    temper.mkdir()
    (temper / surface).mkdir()

    with pytest.raises(EvidenceCorrupt, match="^unsupported_store_surface$"):
        _store(tmp_path).verify()


def test_verification_rejects_reference_logical_id_alias(tmp_path: Path) -> None:
    store = _store(tmp_path)
    task = _task()
    project = Project(
        project_id="project-alias",
        display_name="Synthetic alias project",
        purpose="Reject a logical alias for pinned immutable evidence.",
        task_definition=RecordReference("task_definition", "task-alias", task.identity),
    )
    store.write_record(task)
    store.write_record(project)

    with pytest.raises(EvidenceCorrupt, match="^record_reference_logical_id_mismatch$"):
        store.verify()


def test_reference_shaped_arbitrary_json_is_not_typed_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    task = _task(
        input_schema={
            "record_type": "artifact",
            "logical_id": "schema-property",
            "identity": {
                "algorithm": "sha256",
                "value": "d" * 64,
            },
        }
    )
    store.write_record(task)

    assert store.verify().record_count == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "payload",
        "claimed_identity",
        "schema_version",
        "projection_version",
        "duplicate_key",
        "invalid_utf8",
        "noncanonical",
    ],
)
def test_record_verification_rejects_tampered_bytes(
    tmp_path: Path, mutation: str
) -> None:
    store = _store(tmp_path)
    stored = store.write_record(_task())
    value = loads_canonical_json(stored.path.read_bytes())
    assert isinstance(value, dict)

    if mutation == "payload":
        value["payload"]["description"] = "Changed"
        payload = dumps_canonical_json(value)
    elif mutation == "claimed_identity":
        value["identity"]["value"] = "0" * 64
        payload = dumps_canonical_json(value)
    elif mutation == "schema_version":
        value["schema_version"] = "v2"
        payload = dumps_canonical_json(value)
    elif mutation == "projection_version":
        value["projection_version"] = "v2"
        payload = dumps_canonical_json(value)
    elif mutation == "duplicate_key":
        payload = b'{"record_type":"task_definition","record_type":"project"}'
    elif mutation == "invalid_utf8":
        payload = b"\xff"
    else:
        payload = stored.path.read_bytes() + b"\n"
    stored.path.write_bytes(payload)

    with pytest.raises(EvidenceCorrupt):
        store.iter_records()


def test_record_verification_binds_filename_and_type_directory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stored = store.write_record(_task())
    wrong_name = stored.path.with_name(f"{'0' * 64}.json")
    stored.path.rename(wrong_name)

    with pytest.raises(EvidenceCorrupt, match="^record_identity_path_mismatch$"):
        store.iter_records()

    wrong_name.rename(stored.path)
    wrong_directory = store.layout.record_directory("project")
    wrong_directory.mkdir(parents=True)
    moved = wrong_directory / stored.path.name
    stored.path.rename(moved)

    with pytest.raises(EvidenceCorrupt, match="^record_type_path_mismatch$"):
        store.iter_records()


def test_precise_interrupted_temp_is_ignored_but_unknown_entries_fail(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    stored = store.write_record(_task())
    temporary = stored.path.with_name(f".{stored.path.name}.{'a' * 32}.tmp")
    temporary.write_bytes(b"partial")

    assert len(store.iter_records()) == 1

    temporary.rename(temporary.with_name("unexpected.tmp"))
    with pytest.raises(EvidenceCorrupt, match="^invalid_record_filename$"):
        store.iter_records()


def test_linked_temporary_entry_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stored = store.write_record(_task())
    source = tmp_path / "source"
    source.write_bytes(b"partial")
    linked = stored.path.with_name(f".{stored.path.name}.{'b' * 32}.tmp")
    try:
        os.symlink(source, linked)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(EvidenceCorrupt, match="^unsafe_record_store$"):
        store.iter_records()


def test_typed_store_rejects_linked_immutable_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    temper = tmp_path / ".temper"
    temper.mkdir()
    try:
        os.symlink(outside, temper / "immutable", target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(EvidenceCorrupt, match="^unsafe_evidence_path$"):
        _store(tmp_path).write_record(_task())


def test_admission_rejects_nested_secrets_and_absolute_paths(tmp_path: Path) -> None:
    store = _store(tmp_path)
    secret = _task(input_schema={"properties": {"api_token": {"type": "string"}}})
    path = _task(
        task_id="task-path",
        description="Read C:\\private\\synthetic.txt during preprocessing.",
    )

    with pytest.raises(EvidenceError, match="^admission_secret_field$") as secret_error:
        store.write_record(secret)
    with pytest.raises(EvidenceError, match="^admission_absolute_path$") as path_error:
        store.write_record(path)
    assert "api_token" not in str(secret_error.value)
    assert "private" not in str(path_error.value)


def test_admission_allows_noncredential_ml_token_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    task = _task(
        input_schema={
            "properties": {
                "token": {"type": "string"},
                "token_count": {"type": "integer"},
            }
        }
    )

    assert store.write_record(task).record == task


def test_canonical_reads_and_public_dumps_are_host_context_independent(
    tmp_path: Path,
) -> None:
    writer = _store(tmp_path)
    writer.write_record(
        _task(description="Synthetic-user validates portable canonical evidence.")
    )
    host_aware_reader = TypedEvidenceStore(
        tmp_path,
        redaction_context=RedactionContext(
            local_usernames=("synthetic-user",),
            local_hostnames=("synthetic-host",),
        ),
    )

    assert host_aware_reader.verify().record_count == 1
    assert dumps_canonical_json(host_aware_reader.public_dump().value) == (
        dumps_canonical_json(writer.public_dump().value)
    )


def test_store_admission_never_depends_on_a_runtime_url_allowlist(
    tmp_path: Path,
) -> None:
    store = TypedEvidenceStore(
        tmp_path,
        redaction_context=RedactionContext(
            allowed_public_url_prefixes=("https://docs.example.invalid/",)
        ),
    )
    task = _task(
        description="See https://docs.example.invalid/public for synthetic details."
    )

    with pytest.raises(EvidenceError, match="^admission_url$"):
        store.write_record(task)


def test_bundle_and_file_evidence_are_verified_inside_project(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bundle = tmp_path / "synthetic-bundle"
    bundle.mkdir()
    member = bundle / "weights.bin"
    member.write_bytes(b"synthetic weights")
    manifest = build_bundle_manifest(bundle)

    stored_path = store.write_bundle_manifest(manifest)

    assert stored_path.name == f"{manifest.identity.value}.json"
    assert store.read_bundle_manifest(manifest.identity) == manifest
    store.verify_file_evidence(
        member.relative_to(tmp_path), byte_identity(member.read_bytes())
    )
    store.verify_bundle_evidence(bundle.relative_to(tmp_path), manifest.identity)
    assert store.verify().bundle_manifest_count == 1

    member.write_bytes(b"tampered")
    with pytest.raises(EvidenceCorrupt, match="^file_evidence_mismatch$"):
        store.verify_file_evidence(
            member.relative_to(tmp_path), byte_identity(b"synthetic weights")
        )
    with pytest.raises(EvidenceCorrupt, match="^bundle_evidence_mismatch$"):
        store.verify_bundle_evidence(bundle.relative_to(tmp_path), manifest.identity)
    with pytest.raises(EvidenceError, match="^evidence_path_outside_project$"):
        store.verify_file_evidence(tmp_path.parent / "outside", _identity("outside"))

    (store.layout.bundle_manifests_root().parent / "sha512").mkdir()
    with pytest.raises(EvidenceCorrupt, match="^invalid_bundle_algorithm_directory$"):
        store.iter_bundle_manifests()


def test_manifest_selection_requires_an_exact_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _task(description="First synthetic revision.")
    second = _task(description="Second synthetic revision.")
    store.write_record(first)
    store.write_record(second)

    with pytest.raises(EvidenceAmbiguous, match="^manifest_ambiguous$"):
        store.inspect_manifest("task_definition", first.task_id)
    assert (
        store.inspect_manifest("task_definition", first.task_id, first.identity)
        == first.to_envelope()
    )


def test_rebuild_and_public_dump_preserve_canonical_evidence(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    dump_bytes: list[bytes] = []
    for index, root in enumerate(roots):
        store = _store(root)
        task = _task()
        project = _project(task)
        records = (task, project) if index == 0 else (project, task)
        for record in records:
            store.write_record(record)
        first_event = store.append_event(
            "project-lifecycle",
            EventRequest("event-1", "project_created", {"synthetic": True}),
        )
        second_event = store.append_event(
            "project-lifecycle",
            EventRequest("event-2", "project_verified", {"count": 2}),
        )
        canonical_before = {
            path: path.read_bytes()
            for path in (
                *(stored.path for stored in store.iter_records()),
                first_event.path,
                second_event.path,
            )
        }

        state = store.rebuild_stream_state("project-lifecycle")
        assert state["event_count"] == 2
        state_path = store.layout.stream_state("project-lifecycle")
        state_path.write_bytes(b"corrupt derived state")
        assert store.rebuild_stream_state("project-lifecycle") == state
        state_path.unlink()
        assert store.rebuild_stream_state("project-lifecycle") == state
        assert {
            path: path.read_bytes() for path in canonical_before
        } == canonical_before

        public = store.public_dump()
        encoded = dumps_canonical_json(public.value)
        assert task.task_id.encode() not in encoded
        assert task.identity.value.encode() not in encoded
        assert _identity("synthetic-renderer").value.encode() not in encoded
        assert b"Synthetic transformation" not in encoded
        assert b"input_schema" not in encoded
        assert b"project_created" not in encoded
        dump_bytes.append(encoded)

    assert dump_bytes[0] == dump_bytes[1]


def test_event_admission_and_idempotency_errors_are_public_safe(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_event("events", EventRequest("same", "created", {"value": 1}))

    with pytest.raises(EvidenceCorrupt, match="^event_idempotency_conflict$"):
        store.append_event("events", EventRequest("same", "created", {"value": 2}))
    with pytest.raises(EvidenceError, match="^admission_secret_field$"):
        store.append_event(
            "events",
            EventRequest("secret", "created", {"api_key": "inert"}),
        )
    with pytest.raises(EvidenceError, match="^event_typed_evidence_unsupported$"):
        store.append_event(
            "events",
            EventRequest(
                "reference",
                "created",
                {
                    "record": RecordReference(
                        "task_definition", "task-event", _identity("event-task")
                    ).to_dict()
                },
            ),
        )

    EventStream(store.layout.stream_events("untyped")).append(
        EventRequest(
            "raw-reference",
            "created",
            {
                "record": RecordReference(
                    "task_definition", "task-event", _identity("event-task")
                ).to_dict()
            },
        )
    )
    with pytest.raises(EvidenceCorrupt, match="^event_typed_evidence_unsupported$"):
        store.verify()


def test_rebuild_rejects_linked_derived_state_without_touching_target(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.append_event("events", EventRequest("one", "created", {}))
    state = store.layout.stream_state("events")
    state.parent.mkdir(parents=True)
    target = tmp_path / "target-state"
    target.write_bytes(b"unchanged")
    try:
        os.symlink(target, state)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(EvidenceCorrupt, match="^derived_state_rebuild_failed$"):
        store.rebuild_stream_state("events")
    assert target.read_bytes() == b"unchanged"
    with pytest.raises(EvidenceCorrupt, match="^derived_state_not_rebuildable$"):
        store.verify()
