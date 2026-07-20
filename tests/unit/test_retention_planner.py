import os
from pathlib import Path

import pytest

import temper_ml.app_services.retention as retention_module
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.fixture_journey import FixtureJourneyService
from temper_ml.app_services.local_use import LocalUseService
from temper_ml.app_services.retention import (
    ByteClass,
    CleanupImpact,
    RetentionService,
)
from temper_ml.domain.artifacts import Artifact, ArtifactAvailability, AvailabilityState
from temper_ml.domain.records import RecordEnvelope, record_reference
from temper_ml.domain.retention import (
    CleanupObjectStatus,
    CleanupOutcome,
    CleanupReceipt,
)
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore


def _project(tmp_path: Path, *, launch: bool = False) -> FixtureJourneyService:
    journey = FixtureJourneyService(tmp_path)
    journey.setup_project()
    if launch:
        journey.import_dataset()
        journey.resolve_candidates()
        journey.launch_candidates()
    return journey


def _write_debug_file(tmp_path: Path, name: str, payload: bytes) -> Path:
    root = tmp_path / ".temper-fixture-output" / "logs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(payload)
    return path


def test_inventory_defaults_to_full_and_keeps_canonical_store_out_of_scope(
    tmp_path: Path,
) -> None:
    _project(tmp_path, launch=True)

    inventory = RetentionService(tmp_path).inventory()
    view = inventory.to_view()

    assert view["retention_default"] == "full"
    assert inventory.entries
    assert inventory.logical_bytes >= inventory.physical_bytes > 0
    assert {entry.byte_class for entry in inventory.entries} >= {
        ByteClass.CHECKPOINT,
        ByteClass.FINAL_ADAPTER,
        ByteClass.RUNTIME_CONTROL,
    }
    assert any(
        CleanupImpact.RESUMABILITY in entry.impacts
        for entry in inventory.entries
        if entry.byte_class is ByteClass.CHECKPOINT
    )
    assert all(
        not entry.logical_key.startswith(".temper/") for entry in inventory.entries
    )
    assert all(str(tmp_path) not in entry.logical_key for entry in inventory.entries)
    assert all(
        not entry.deletable
        for entry in inventory.entries
        if entry.byte_class is ByteClass.RUNTIME_CONTROL
    )


def test_shared_reference_accounting_counts_only_bytes_that_can_be_freed(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    first = _write_debug_file(tmp_path, "shared-a.bin", b"shared-fixture")
    second = first.with_name("shared-b.bin")
    os.link(first, second)
    outside = tmp_path / "retained-hardlink.bin"
    os.link(first, outside)

    service = RetentionService(tmp_path)
    entries = tuple(
        entry
        for entry in service.inventory().entries
        if entry.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    assert len(entries) == 2
    assert {entry.local_reference_count for entry in entries} == {2}
    assert {entry.external_reference_count for entry in entries} == {1}

    one = service.plan((entries[0].entry_id,))
    both = service.plan(tuple(entry.entry_id for entry in entries))

    assert one.physical_bytes_freed == 0
    assert both.physical_bytes_freed == 0
    assert CleanupImpact.SHARED_REFERENCE in both.impact_categories


def test_complete_internal_hardlink_group_is_removed_as_one_physical_object(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    first = _write_debug_file(tmp_path, "linked-a.bin", b"one physical object")
    second = first.with_name("linked-b.bin")
    os.link(first, second)
    service = RetentionService(tmp_path)
    entries = tuple(
        entry
        for entry in service.inventory().entries
        if entry.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    plan = service.plan(tuple(entry.entry_id for entry in entries))

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.COMPLETED
    assert not first.exists()
    assert not second.exists()
    assert receipt.logical_bytes_removed == 2 * len(b"one physical object")
    assert receipt.physical_bytes_freed == len(b"one physical object")
    assert all(item.status is CleanupObjectStatus.REMOVED for item in receipt.objects)


def test_only_terminal_run_staging_is_cleanup_eligible_and_warns_about_cache_loss(
    tmp_path: Path,
) -> None:
    _project(tmp_path, launch=True)
    staging = tmp_path / ".temper-fixture-output" / "library-staging"
    terminal = staging / "run-fixture-runtime" / "host-to-worker" / "cache.bin"
    unbound = staging / "unbound-run" / "host-to-worker" / "cache.bin"
    for path in (terminal, unbound):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic staging cache")

    service = RetentionService(tmp_path)
    entries = {
        entry.logical_key: entry
        for entry in service.inventory().entries
        if entry.byte_class is ByteClass.STAGING_CACHE
    }

    assert entries[
        "library-staging/run-fixture-runtime/host-to-worker/cache.bin"
    ].deletable
    assert not entries["library-staging/unbound-run/host-to-worker/cache.bin"].deletable
    plan = service.plan(
        (
            entries[
                "library-staging/run-fixture-runtime/host-to-worker/cache.bin"
            ].entry_id,
        )
    )
    assert CleanupImpact.CACHE_CONVENIENCE in plan.impact_categories
    assert {warning["category"] for warning in plan.to_view()["warnings"]} >= {
        "cache_convenience",
        "debugging_evidence",
    }


def test_cleanup_removes_selected_artifact_bytes_and_supersedes_availability(
    tmp_path: Path,
) -> None:
    _project(tmp_path, launch=True)
    store = TypedEvidenceStore(tmp_path)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / ".temper").rglob("*")
        if path.is_file()
    }
    service = RetentionService(tmp_path)
    selected = tuple(
        entry.entry_id
        for entry in service.inventory().entries
        if entry.byte_class is ByteClass.FINAL_ADAPTER
        and any(
            subject.logical_id == "artifact-fixture-runtime"
            for subject in entry.subjects
        )
    )
    assert selected

    plan = service.plan(selected)
    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.COMPLETED
    assert receipt.logical_bytes_removed == plan.logical_bytes_selected
    assert receipt.physical_bytes_freed == plan.physical_bytes_freed
    assert all(item.status is CleanupObjectStatus.REMOVED for item in receipt.objects)
    assert RecordEnvelope.from_dict(receipt.to_dict()) == receipt.to_envelope()
    for relative, payload in before.items():
        assert (tmp_path / relative).read_bytes() == payload
    records = tuple(item.record for item in store.iter_records())
    artifact = next(
        item
        for item in records
        if isinstance(item, Artifact) and item.artifact_id == "artifact-fixture-runtime"
    )
    availabilities = tuple(
        item
        for item in records
        if isinstance(item, ArtifactAvailability)
        and item.artifact == record_reference(artifact)
    )
    current = [
        item
        for item in availabilities
        if item.identity
        not in {
            candidate.supersedes.identity
            for candidate in availabilities
            if candidate.supersedes is not None
        }
    ]
    assert len(current) == 1
    assert current[0].state is AvailabilityState.REMOVED
    model = store.read_record(artifact.base_model_revision).record
    group = store.read_record(artifact.compatibility_groups[0]).record
    run = store.read_record(artifact.producing_run).record
    target = store.read_record(run.execution_target).record
    with pytest.raises(
        ApplicationServiceError, match="^local_use_artifact_unavailable$"
    ):
        LocalUseService(tmp_path).inspect_artifact(artifact, model, group, target)
    assert store.verify().to_dict()["status"] == "verified"


def test_cleanup_rejects_a_changed_snapshot_without_removing_bytes(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    path = _write_debug_file(tmp_path, "stale.log", b"before")
    service = RetentionService(tmp_path)
    entry = next(
        item
        for item in service.inventory().entries
        if item.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    plan = service.plan((entry.entry_id,))
    path.write_bytes(b"after")

    with pytest.raises(ApplicationServiceError, match="^cleanup_plan_stale$"):
        service.execute(plan, confirm=True)

    assert path.read_bytes() == b"after"
    assert not service.receipts()


def test_partial_cleanup_records_removed_failed_and_unattempted_objects(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    for name in ("one.log", "two.log", "three.log"):
        _write_debug_file(tmp_path, name, name.encode())
    failure = {"name": ""}

    def remove(path: Path) -> None:
        if path.name == failure["name"]:
            raise PermissionError(path.name)
        os.unlink(path)

    service = RetentionService(tmp_path, _remove_file=remove)
    entries = tuple(
        entry
        for entry in service.inventory().entries
        if entry.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    plan = service.plan(tuple(entry.entry_id for entry in entries))
    failure["name"] = plan.selected_entries[1]._path.name

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.PARTIAL
    assert receipt.failure_code == "cleanup_object_remove_failed"
    assert [item.status for item in receipt.objects] == [
        CleanupObjectStatus.REMOVED,
        CleanupObjectStatus.RETAINED,
        CleanupObjectStatus.NOT_ATTEMPTED,
    ]
    assert isinstance(
        TypedEvidenceStore(tmp_path).read_record(record_reference(receipt)).record,
        CleanupReceipt,
    )


def test_failed_artifact_unlink_restores_verified_availability(
    tmp_path: Path,
) -> None:
    _project(tmp_path, launch=True)

    def retain_file(path: Path) -> None:
        raise PermissionError(path.name)

    service = RetentionService(tmp_path, _remove_file=retain_file)
    artifact = next(
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, Artifact)
        and stored.record.artifact_id == "artifact-fixture-runtime"
    )
    before = _current_artifact_availability(service.store, artifact)
    selected = tuple(
        entry.entry_id
        for entry in service.inventory().entries
        if entry.byte_class is ByteClass.FINAL_ADAPTER
        and record_reference(artifact) in entry.subjects
    )
    plan = service.plan(selected)

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.FAILED
    assert receipt.failure_code == "cleanup_object_remove_failed"
    assert receipt.objects[0].status is CleanupObjectStatus.RETAINED
    assert all(entry._path.exists() for entry in plan.selected_entries)
    current = _current_artifact_availability(service.store, artifact)
    assert current.state is before.state
    assert current.available_byte_classes == before.available_byte_classes
    assert current.storage_references == before.storage_references
    assert current.checkpoint_resumable == before.checkpoint_resumable


def test_artifact_changed_after_safety_fence_remains_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path, launch=True)
    service = RetentionService(tmp_path)
    artifact = next(
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, Artifact)
        and stored.record.artifact_id == "artifact-fixture-runtime"
    )
    selected = tuple(
        entry
        for entry in service.inventory().entries
        if entry.byte_class is ByteClass.FINAL_ADAPTER
        and record_reference(artifact) in entry.subjects
    )
    assert selected
    plan = service.plan(tuple(entry.entry_id for entry in selected))
    original_prepare = service._prepare_safety_fences

    def fence_then_mutate(intent: object) -> None:
        original_prepare(intent)  # type: ignore[arg-type]
        selected[0]._path.write_bytes(b"changed after the availability fence")

    monkeypatch.setattr(service, "_prepare_safety_fences", fence_then_mutate)

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.FAILED
    assert receipt.failure_code == "cleanup_object_changed"
    assert receipt.objects[0].status is CleanupObjectStatus.AMBIGUOUS
    current = _current_artifact_availability(service.store, artifact)
    assert current.state is AvailabilityState.UNAVAILABLE
    assert current.available_byte_classes == ()
    assert current.storage_references == ()
    assert current.checkpoint_resumable is False
    assert selected[0]._path.exists()


def test_failed_checkpoint_unlink_restores_resume_availability(
    tmp_path: Path,
) -> None:
    journey = _project(tmp_path, launch=True)

    def retain_file(path: Path) -> None:
        raise PermissionError(path.name)

    service = RetentionService(tmp_path, _remove_file=retain_file)
    entry = next(
        item
        for item in service.inventory().entries
        if item.byte_class is ByteClass.CHECKPOINT
        and CleanupImpact.RESUMABILITY in item.impacts
    )
    run_id = next(
        subject.logical_id for subject in entry.subjects if subject.record_type == "run"
    )
    before = next(
        item for item in journey.workspace()["runs"] if item["run_id"] == run_id
    )
    plan = service.plan((entry.entry_id,))

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.FAILED
    assert receipt.objects[0].status is CleanupObjectStatus.RETAINED
    assert entry._path.exists()
    after = next(
        item
        for item in FixtureJourneyService(tmp_path).workspace()["runs"]
        if item["run_id"] == run_id
    )
    assert (
        after["resume_available_checkpoint_count"]
        == before["resume_available_checkpoint_count"]
    )
    assert any(
        event["type"] == "run_checkpoint_cleanup_cancelled"
        and event["resume_available"] is True
        for event in after["events"]
    )


def test_corrupted_checkpoint_is_protected_and_not_advertised_as_resumable(
    tmp_path: Path,
) -> None:
    journey = _project(tmp_path, launch=True)
    service = RetentionService(tmp_path)
    checkpoint = next(
        item
        for item in service.inventory().entries
        if item.byte_class is ByteClass.CHECKPOINT
        and CleanupImpact.RESUMABILITY in item.impacts
    )
    run_id = next(
        subject.logical_id
        for subject in checkpoint.subjects
        if subject.record_type == "run"
    )
    before = next(
        item for item in journey.workspace()["runs"] if item["run_id"] == run_id
    )
    checkpoint._path.write_bytes(b"corrupted synthetic checkpoint")

    corrupted = next(
        item
        for item in service.inventory().entries
        if item.logical_key == checkpoint.logical_key
    )

    assert corrupted.byte_class is ByteClass.UNKNOWN
    assert corrupted.deletable is False
    with pytest.raises(ApplicationServiceError, match="^cleanup_selection_protected$"):
        service.plan((corrupted.entry_id,))
    after = next(
        item
        for item in FixtureJourneyService(tmp_path).workspace()["runs"]
        if item["run_id"] == run_id
    )
    assert (
        after["resume_available_checkpoint_count"]
        == before["resume_available_checkpoint_count"] - 1
    )
    assert checkpoint._path.exists()


def test_inventory_and_removal_use_bounded_streaming_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path)
    path = _write_debug_file(tmp_path, "streamed.log", b"streamed-heavy-bytes")
    observed_chunks: list[int] = []
    original_snapshot = retention_module._stream_file_snapshot

    def observe_snapshot(path: Path, *, chunk_size: int):
        observed_chunks.append(chunk_size)
        return original_snapshot(path, chunk_size=chunk_size)

    monkeypatch.setattr(retention_module, "_stream_file_snapshot", observe_snapshot)
    service = RetentionService(tmp_path, _hash_chunk_size=3)
    entry = next(
        item
        for item in service.inventory().entries
        if item.logical_key == "logs/streamed.log"
    )
    plan = service.plan((entry.entry_id,))

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.COMPLETED
    assert not path.exists()
    assert len(observed_chunks) >= 4
    assert set(observed_chunks) == {3}


def test_checkpoint_event_failure_stays_fenced_and_reconciles_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path, launch=True)
    service = RetentionService(tmp_path)
    entry = next(
        item
        for item in service.inventory().entries
        if item.byte_class is ByteClass.CHECKPOINT
        and CleanupImpact.RESUMABILITY in item.impacts
    )
    plan = service.plan((entry.entry_id,))
    original_append = service.store.append_event
    failed = False

    def fail_removed_event(stream_id: str, request: object) -> object:
        nonlocal failed
        if (
            getattr(request, "event_type", None) == "run_checkpoint_removed"
            and not failed
        ):
            failed = True
            raise EvidenceError("fixture_checkpoint_event_failed")
        return original_append(stream_id, request)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "append_event", fail_removed_event)

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.PARTIAL
    assert receipt.failure_code == "fixture_checkpoint_event_failed"
    assert not entry._path.exists()
    pending_events = next(
        snapshot.events
        for snapshot in service.store.iter_streams()
        if snapshot.stream_id == f"run-{entry.subjects[0].logical_id}"
    )
    assert any(
        event.event_type == "run_checkpoint_cleanup_pending" for event in pending_events
    )

    RetentionService(tmp_path).inventory()
    workspace = FixtureJourneyService(tmp_path).workspace()
    run_view = next(
        item
        for item in workspace["runs"]
        if item["run_id"] == entry.subjects[0].logical_id
    )
    assert run_view["resume_available_checkpoint_count"] == 0
    reconciled_events = next(
        snapshot.events
        for snapshot in TypedEvidenceStore(tmp_path).iter_streams()
        if snapshot.stream_id == f"run-{entry.subjects[0].logical_id}"
    )
    assert any(
        event.event_type == "run_checkpoint_removed" for event in reconciled_events
    )


def test_artifact_availability_write_failure_remains_unavailable_then_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path, launch=True)
    service = RetentionService(tmp_path)
    selected = tuple(
        item.entry_id
        for item in service.inventory().entries
        if item.byte_class is ByteClass.FINAL_ADAPTER
        and any(
            subject.logical_id == "artifact-fixture-runtime"
            for subject in item.subjects
        )
    )
    plan = service.plan(selected)
    original_write = retention_module.write_record_idempotently
    failed = False

    def fail_removed_availability(*args: object, **kwargs: object) -> object:
        nonlocal failed
        record = args[1]
        if (
            isinstance(record, ArtifactAvailability)
            and record.state is AvailabilityState.REMOVED
            and not failed
        ):
            failed = True
            raise EvidenceError("fixture_availability_write_failed")
        return original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        retention_module, "write_record_idempotently", fail_removed_availability
    )

    receipt = service.execute(plan, confirm=True)

    assert receipt.outcome is CleanupOutcome.PARTIAL
    assert receipt.failure_code == "fixture_availability_write_failed"
    artifact = next(
        item.record
        for item in service.store.iter_records()
        if isinstance(item.record, Artifact)
        and item.record.artifact_id == "artifact-fixture-runtime"
    )
    pending = _current_artifact_availability(service.store, artifact)
    assert pending.state is AvailabilityState.UNAVAILABLE

    monkeypatch.setattr(retention_module, "write_record_idempotently", original_write)
    RetentionService(tmp_path).inventory()
    current = _current_artifact_availability(TypedEvidenceStore(tmp_path), artifact)
    assert current.state is AvailabilityState.REMOVED


def test_receipt_write_failure_reconciles_from_durable_intent_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path)
    path = _write_debug_file(tmp_path, "receipt-failure.log", b"receipt recovery")
    service = RetentionService(tmp_path)
    entry = next(
        item
        for item in service.inventory().entries
        if item.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    plan = service.plan((entry.entry_id,))
    original_write = retention_module.write_record_idempotently

    def fail_receipt(*args: object, **kwargs: object) -> object:
        if isinstance(args[1], CleanupReceipt):
            raise EvidenceError("fixture_receipt_write_failed")
        return original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(retention_module, "write_record_idempotently", fail_receipt)

    with pytest.raises(
        ApplicationServiceError, match="^cleanup_reconciliation_required$"
    ):
        service.execute(plan, confirm=True)

    assert not path.exists()
    assert not any(
        isinstance(item.record, CleanupReceipt) for item in service.store.iter_records()
    )
    monkeypatch.setattr(retention_module, "write_record_idempotently", original_write)

    receipts = RetentionService(tmp_path).receipts()

    assert len(receipts) == 1
    assert receipts[0].execution_id == plan.execution_id
    assert receipts[0].outcome is CleanupOutcome.PARTIAL
    assert receipts[0].failure_code == "fixture_receipt_write_failed"
    assert receipts[0].objects[0].status is CleanupObjectStatus.REMOVED


def test_unlink_success_with_ambiguous_error_defers_receipt_to_reconciliation(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    path = _write_debug_file(tmp_path, "ambiguous-unlink.log", b"ambiguous unlink")

    def remove_then_report_failure(target: Path) -> None:
        os.unlink(target)
        raise PermissionError("synthetic post-unlink failure")

    service = RetentionService(tmp_path, _remove_file=remove_then_report_failure)
    entry = next(
        item
        for item in service.inventory().entries
        if item.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    plan = service.plan((entry.entry_id,))

    with pytest.raises(
        ApplicationServiceError, match="^cleanup_reconciliation_required$"
    ):
        service.execute(plan, confirm=True)

    assert not path.exists()
    assert not any(
        isinstance(item.record, CleanupReceipt) for item in service.store.iter_records()
    )

    receipts = RetentionService(tmp_path).receipts()
    assert len(receipts) == 1
    assert receipts[0].execution_id == plan.execution_id
    assert receipts[0].outcome is CleanupOutcome.PARTIAL
    assert receipts[0].failure_code == "cleanup_object_removal_ambiguous"
    assert receipts[0].objects[0].status is CleanupObjectStatus.REMOVED


def test_recreated_identical_bytes_receive_a_distinct_cleanup_execution(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    path = _write_debug_file(tmp_path, "repeat.log", b"repeatable bytes")
    first_service = RetentionService(tmp_path)
    first_entry = next(
        item
        for item in first_service.inventory().entries
        if item.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    first_plan = first_service.plan((first_entry.entry_id,))
    first_receipt = first_service.execute(first_plan, confirm=True)
    path.write_bytes(b"repeatable bytes")

    second_service = RetentionService(tmp_path)
    second_entry = next(
        item
        for item in second_service.inventory().entries
        if item.byte_class is ByteClass.DEBUG_EVIDENCE
    )
    second_plan = second_service.plan((second_entry.entry_id,))
    second_receipt = second_service.execute(second_plan, confirm=True)

    assert first_plan.plan_identity == second_plan.plan_identity
    assert first_plan.execution_id != second_plan.execution_id
    assert first_receipt.receipt_id != second_receipt.receipt_id
    assert {item.receipt_id for item in second_service.receipts()} == {
        first_receipt.receipt_id,
        second_receipt.receipt_id,
    }


def test_parallel_cleanup_execution_is_rejected_before_evidence_can_fork(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    _write_debug_file(tmp_path, "serialized.log", b"serialized cleanup")
    observed_codes: list[str] = []
    second_service = RetentionService(tmp_path)
    second_plan = second_service.plan(
        (
            next(
                item
                for item in second_service.inventory().entries
                if item.byte_class is ByteClass.DEBUG_EVIDENCE
            ).entry_id,
        )
    )

    def remove_while_competing(target: Path) -> None:
        try:
            second_service.execute(second_plan, confirm=True)
        except ApplicationServiceError as exc:
            observed_codes.append(exc.code)
        else:
            observed_codes.append("cleanup_competitor_unexpectedly_executed")
        os.unlink(target)

    first_service = RetentionService(tmp_path, _remove_file=remove_while_competing)
    first_plan = first_service.plan(
        (
            next(
                item
                for item in first_service.inventory().entries
                if item.byte_class is ByteClass.DEBUG_EVIDENCE
            ).entry_id,
        )
    )

    first_receipt = first_service.execute(first_plan, confirm=True)
    receipts = RetentionService(tmp_path).receipts()

    assert observed_codes == ["cleanup_execution_busy"]
    assert tuple(item.receipt_id for item in receipts) == (first_receipt.receipt_id,)
    assert receipts[0].outcome is CleanupOutcome.COMPLETED


def _current_artifact_availability(
    store: TypedEvidenceStore, artifact: Artifact
) -> ArtifactAvailability:
    values = tuple(
        item.record
        for item in store.iter_records()
        if isinstance(item.record, ArtifactAvailability)
        and item.record.artifact == record_reference(artifact)
    )
    superseded = {
        item.supersedes.identity for item in values if item.supersedes is not None
    }
    current = tuple(item for item in values if item.identity not in superseded)
    assert len(current) == 1
    return current[0]
