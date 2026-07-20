import json
from pathlib import Path

import pytest

from temper_ml.cli import main
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.fixture_journey import FixtureJourneyService
from temper_ml.app_services.reproduction import (
    ReplayExecutionRequest,
    ReproductionService,
)
from temper_ml.app_services.retention import (
    ByteClass,
    CleanupImpact,
    RetentionService,
)
from temper_ml.app_services.runs import RunLifecycleStatus, RunService
from temper_ml.domain.retention import CleanupOutcome
from temper_ml.runtime.fixture_adapter import FixtureControl
from temper_ml.store.evidence import EvidenceError, TypedEvidenceStore


def _launched_journey(tmp_path: Path) -> FixtureJourneyService:
    journey = FixtureJourneyService(tmp_path)
    journey.setup_project()
    journey.import_dataset()
    journey.resolve_candidates()
    journey.launch_candidates()
    return journey


def test_strict_and_adapted_replay_execute_as_distinct_new_runs(
    tmp_path: Path,
    capsys,
) -> None:
    journey = _launched_journey(tmp_path)
    journey = FixtureJourneyService(tmp_path)

    strict = journey.prepare_replay("ember", "strict_replay")

    assert strict["status"] == "ready"
    assert strict["mode"] == "strict_replay"
    assert strict["source_manifest_identity"] == strict["planned_manifest_identity"]
    assert strict["manifest_changes"] == []
    with pytest.raises(
        ApplicationServiceError, match="^replay_candidate_plan_mismatch$"
    ):
        journey.execute_replay(
            strict["plan_id"],
            run_id=strict["run_id"],
            candidate_key="slate",
            mode="strict_replay",
        )
    strict_result = journey.execute_replay(strict["plan_id"], run_id=strict["run_id"])
    assert strict_result["status"] == "completed"
    assert strict_result["exact_reproduction"] is True
    assert strict_result["adapted_reproduction"] is False

    adapted = journey.prepare_replay("ember", "adapted_reproduction")

    assert adapted["status"] == "ready"
    assert adapted["mode"] == "adapted_reproduction"
    assert adapted["source_manifest_identity"] != adapted["planned_manifest_identity"]
    changed_roots = {
        change["path"].split("/", 2)[1] for change in adapted["manifest_changes"]
    }
    assert changed_roots == {"hardware_requirements", "recipe_resolution"}
    adapted_result = journey.execute_replay(
        adapted["plan_id"], run_id=adapted["run_id"]
    )
    assert adapted_result["status"] == "completed"
    assert adapted_result["exact_reproduction"] is False
    assert adapted_result["adapted_reproduction"] is True

    workspace = journey.workspace()
    assert {item["mode"] for item in workspace["reproduction"]["executions"]} == {
        "strict_replay",
        "adapted_reproduction",
    }
    assert len(workspace["reproduction"]["derivations"]) == 1
    assert TypedEvidenceStore(tmp_path).verify().to_dict()["status"] == "verified"

    estimate_arguments = [
        "--base-model-bytes",
        "0",
        "--adapter-optimizer-bytes",
        "0",
        "--peak-activation-bytes",
        "0",
        "--accelerator-runtime-overhead-bytes",
        "0",
        "--dataset-bytes",
        "0",
        "--host-runtime-overhead-bytes",
        "0",
    ]
    assert (
        main(
            [
                "replay-plan",
                str(tmp_path),
                "--experiment-id",
                "experiment-fixture-runtime",
                "--profile-id",
                "profile-fixture-runtime",
                *estimate_arguments,
            ]
        )
        == 0
    )
    strict_cli = json.loads(capsys.readouterr().out)
    assert strict_cli["mode"] == "strict_replay"
    assert strict_cli["status"] == "ready"

    assert (
        main(
            [
                "replay-plan",
                str(tmp_path),
                "--experiment-id",
                "experiment-fixture-runtime",
                "--profile-id",
                "profile-replay-ember-002",
                "--mode",
                "adapted_reproduction",
                "--derivation-id",
                "derivation-adapted-001",
                *estimate_arguments,
            ]
        )
        == 0
    )
    adapted_cli = json.loads(capsys.readouterr().out)
    assert adapted_cli["mode"] == "adapted_reproduction"
    assert adapted_cli["status"] == "ready"
    assert adapted_cli["manifest_changes"]


def test_checkpoint_cleanup_preserves_existing_canonical_bytes_and_removes_resume(
    tmp_path: Path,
) -> None:
    journey = _launched_journey(tmp_path)
    service = RetentionService(tmp_path)
    checkpoint = next(
        entry
        for entry in service.inventory().entries
        if CleanupImpact.RESUMABILITY in entry.impacts
    )
    run_id = next(
        subject.logical_id
        for subject in checkpoint.subjects
        if subject.record_type == "run"
    )
    before_workspace = journey.workspace()
    before_run = next(
        run for run in before_workspace["runs"] if run["run_id"] == run_id
    )
    assert before_run["resume_available_checkpoint_count"] > 0
    existing_canonical = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / ".temper").rglob("*")
        if path.is_file()
    }

    plan = journey.preview_cleanup((checkpoint.entry_id,))

    assert plan["requires_confirmation"] is True
    assert {warning["category"] for warning in plan["warnings"]} >= {
        "resumability",
        "inspectability",
        "debugging_evidence",
    }
    with pytest.raises(
        ApplicationServiceError, match="^cleanup_selection_plan_mismatch$"
    ):
        journey.execute_cleanup(
            plan["plan_id"], confirm=True, entry_ids=("entry-not-in-plan",)
        )
    receipt = journey.execute_cleanup(plan["plan_id"], confirm=True)

    assert receipt["outcome"] == CleanupOutcome.COMPLETED.value
    assert receipt["physical_bytes_freed"] > 0
    assert not checkpoint._path.exists()
    for relative, payload in existing_canonical.items():
        assert (tmp_path / relative).read_bytes() == payload
    after_workspace = journey.workspace()
    after_run = next(run for run in after_workspace["runs"] if run["run_id"] == run_id)
    assert (
        after_run["resume_available_checkpoint_count"]
        == before_run["resume_available_checkpoint_count"] - 1
    )
    assert any(
        event["type"] == "run_checkpoint_removed" and event["resume_available"] is False
        for event in after_run["events"]
    )
    assert RunService(tmp_path).status(run_id) is RunLifecycleStatus.COMPLETED
    assert len(after_workspace["retention"]["receipts"]) == 1


@pytest.mark.parametrize("after_retention", [False, True], ids=["retained", "cleaned"])
def test_replay_completion_reconciles_after_post_run_event_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_retention: bool,
) -> None:
    journey = _launched_journey(tmp_path)
    strict = journey.prepare_replay("ember", "strict_replay")
    draft = journey._replay_draft
    assert draft is not None
    plan_id = draft.plan.plan_id
    run_id = draft.launch.run_id
    artifact_id = draft.launch.artifact_id
    request = ReplayExecutionRequest(draft.plan, draft.launch, draft.candidate_key)
    service = ReproductionService(tmp_path)
    original_append = service.store.append_event
    failed = False

    def fail_completed_event(stream_id: str, event_request: object) -> object:
        nonlocal failed
        if (
            getattr(event_request, "event_type", None) == "replay_execution_completed"
            and not failed
        ):
            failed = True
            raise EvidenceError("fixture_replay_completion_failed")
        return original_append(stream_id, event_request)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "append_event", fail_completed_event)

    with pytest.raises(
        ApplicationServiceError, match="^fixture_replay_completion_failed$"
    ):
        service.execute(request)

    run_stream_id = f"run-{run_id}"
    replay_stream = next(
        snapshot
        for snapshot in TypedEvidenceStore(tmp_path).iter_streams()
        if snapshot.stream_id.startswith("replay-")
        and any(event.payload.get("run_id") == run_id for event in snapshot.events)
    )
    assert [event.event_type for event in replay_stream.events] == [
        "replay_execution_started"
    ]

    del request, draft, service, journey

    if after_retention:
        retention = RetentionService(tmp_path)
        selected = tuple(
            entry
            for entry in retention.inventory().entries
            if (
                entry.byte_class is ByteClass.CHECKPOINT
                and any(
                    subject.record_type == "run" and subject.logical_id == run_id
                    for subject in entry.subjects
                )
            )
            or (
                entry.byte_class is ByteClass.FINAL_ADAPTER
                and any(
                    subject.record_type == "artifact"
                    and subject.logical_id == artifact_id
                    for subject in entry.subjects
                )
            )
        )
        assert {entry.byte_class for entry in selected} == {
            ByteClass.CHECKPOINT,
            ByteClass.FINAL_ADAPTER,
        }
        cleanup = retention.execute(
            retention.plan(tuple(entry.entry_id for entry in selected)),
            confirm=True,
        )
        assert cleanup.outcome is CleanupOutcome.COMPLETED
        assert all(not entry._path.exists() for entry in selected)

    restarted = FixtureJourneyService(tmp_path)
    before_reconciliation = restarted.workspace()
    pending_execution = next(
        item
        for item in before_reconciliation["reproduction"]["executions"]
        if item["run_id"] == run_id
    )
    assert pending_execution["status"] == "running"
    reconciled = restarted.execute_replay(
        plan_id,
        run_id=run_id,
        candidate_key="ember",
        mode="strict_replay",
    )

    assert reconciled["status"] == "completed"
    assert reconciled["reconciled"] is True
    assert reconciled["execution"]["run_id"] == run_id
    workspace = restarted.workspace()
    execution = next(
        item
        for item in workspace["reproduction"]["executions"]
        if item["run_id"] == run_id
    )
    assert execution["status"] == "completed"
    replay_stream = next(
        snapshot
        for snapshot in TypedEvidenceStore(tmp_path).iter_streams()
        if snapshot.stream_id == replay_stream.stream_id
    )
    assert [event.event_type for event in replay_stream.events] == [
        "replay_execution_started",
        "replay_execution_completed",
    ]
    run_events = next(
        snapshot.events
        for snapshot in TypedEvidenceStore(tmp_path).iter_streams()
        if snapshot.stream_id == run_stream_id
    )
    assert sum(event.event_type == "run_launched" for event in run_events) == 1
    assert sum(event.event_type == "run_completed" for event in run_events) == 1
    assert strict["plan_id"] == plan_id
    assert TypedEvidenceStore(tmp_path).verify().to_dict()["status"] == "verified"


@pytest.mark.parametrize(
    "failed_event_type",
    (
        "run_preflight_succeeded",
        "runtime_request_frozen",
        "run_launched",
    ),
)
def test_replay_launch_prefix_failure_reopens_with_truthful_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_event_type: str,
) -> None:
    journey = _launched_journey(tmp_path)
    prepared = journey.prepare_replay("ember", "strict_replay")
    draft = journey._replay_draft
    assert draft is not None
    request = ReplayExecutionRequest(draft.plan, draft.launch, draft.candidate_key)
    run_id = draft.launch.run_id
    failure_code = f"fixture_{failed_event_type}_before_commit"
    original_append = TypedEvidenceStore.append_event
    failed = False

    def fail_prefix_once(
        store: TypedEvidenceStore, stream_id: str, event_request: object
    ) -> object:
        nonlocal failed
        if (
            stream_id == f"run-{run_id}"
            and getattr(event_request, "event_type", None) == failed_event_type
            and not failed
        ):
            failed = True
            raise EvidenceError(failure_code)
        return original_append(store, stream_id, event_request)  # type: ignore[arg-type]

    monkeypatch.setattr(TypedEvidenceStore, "append_event", fail_prefix_once)

    with pytest.raises(ApplicationServiceError, match=f"^{failure_code}$"):
        ReproductionService(tmp_path).execute(request)

    del request, draft, journey
    store = TypedEvidenceStore(tmp_path)
    run_events = next(
        snapshot.events
        for snapshot in store.iter_streams()
        if snapshot.stream_id == f"run-{run_id}"
    )
    assert sum(event.event_type == "run_failed" for event in run_events) == 1
    assert not any(event.event_type == "run_launched" for event in run_events)
    replay_events = next(
        snapshot.events
        for snapshot in store.iter_streams()
        if snapshot.stream_id.startswith("replay-")
        and any(event.payload.get("run_id") == run_id for event in snapshot.events)
    )
    assert [event.event_type for event in replay_events] == [
        "replay_execution_started",
        "replay_execution_failed",
    ]

    restarted = FixtureJourneyService(tmp_path)
    workspace = restarted.workspace()
    run_view = next(item for item in workspace["runs"] if item["run_id"] == run_id)
    assert run_view["status"] == "failed"
    reconciled = restarted.execute_replay(
        prepared["plan_id"],
        run_id=run_id,
        candidate_key="ember",
        mode="strict_replay",
    )
    assert reconciled["status"] == "failed"
    assert reconciled["execution"]["run_id"] == run_id
    assert reconciled["execution"]["failure_code"] == failure_code
    assert store.verify().to_dict()["status"] == "verified"


def test_identical_replay_plans_reconcile_the_exact_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journey = _launched_journey(tmp_path)
    first = journey.prepare_replay("ember", "strict_replay")
    first_result = journey.execute_replay(first["plan_id"], run_id=first["run_id"])
    assert first_result["status"] == "completed"

    second = journey.prepare_replay("ember", "strict_replay")
    draft = journey._replay_draft
    assert draft is not None
    assert second["plan_id"] == first["plan_id"]
    assert second["run_id"] != first["run_id"]
    request = ReplayExecutionRequest(draft.plan, draft.launch, draft.candidate_key)
    service = ReproductionService(tmp_path)
    original_append = service.store.append_event
    failed = False

    def lose_second_terminal(stream_id: str, event_request: object) -> object:
        nonlocal failed
        if (
            getattr(event_request, "event_type", None) == "replay_execution_completed"
            and not failed
        ):
            failed = True
            raise EvidenceError("fixture_second_replay_terminal_lost")
        return original_append(stream_id, event_request)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "append_event", lose_second_terminal)
    with pytest.raises(
        ApplicationServiceError, match="^fixture_second_replay_terminal_lost$"
    ):
        service.execute(request)

    del request, draft, service, journey
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / ".temper").rglob("*")
        if path.is_file()
    }
    restarted = FixtureJourneyService(tmp_path)
    workspace = restarted.workspace()
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / ".temper").rglob("*")
        if path.is_file()
    }
    assert after == before
    statuses = {
        item["run_id"]: item["status"]
        for item in workspace["reproduction"]["executions"]
    }
    assert statuses[first["run_id"]] == "completed"
    assert statuses[second["run_id"]] == "running"

    reconciled = restarted.execute_replay(
        second["plan_id"],
        run_id=second["run_id"],
        candidate_key="ember",
        mode="strict_replay",
    )

    assert reconciled["status"] == "completed"
    assert reconciled["execution"]["run_id"] == second["run_id"]
    replay_streams = tuple(
        snapshot
        for snapshot in TypedEvidenceStore(tmp_path).iter_streams()
        if snapshot.stream_id.startswith("replay-")
    )
    assert len(replay_streams) == 2
    assert all(
        sum(
            event.event_type == "replay_execution_completed"
            for event in snapshot.events
        )
        == 1
        for snapshot in replay_streams
    )


@pytest.mark.parametrize(
    ("control", "expected_status", "terminal_event"),
    [
        (
            FixtureControl(cancel_after_step=2),
            RunLifecycleStatus.CANCELLED,
            "replay_execution_cancelled",
        ),
        (
            FixtureControl(interrupt_after_step=3),
            RunLifecycleStatus.INTERRUPTED,
            "replay_execution_interrupted",
        ),
    ],
    ids=["cancelled", "interrupted"],
)
def test_replay_records_the_actual_noncompleted_terminal_outcome(
    tmp_path: Path,
    control: FixtureControl,
    expected_status: RunLifecycleStatus,
    terminal_event: str,
) -> None:
    journey = _launched_journey(tmp_path)
    prepared = journey.prepare_replay("ember", "strict_replay")
    draft = journey._replay_draft
    assert draft is not None
    plan_id = draft.plan.plan_id
    run_id = draft.launch.run_id

    result = ReproductionService(tmp_path).execute(
        ReplayExecutionRequest(draft.plan, draft.launch, draft.candidate_key),
        control=control,
    )

    assert result.to_view()["status"] == expected_status.value
    del result, draft, journey
    restarted = FixtureJourneyService(tmp_path)
    workspace = restarted.workspace()
    execution = next(
        item
        for item in workspace["reproduction"]["executions"]
        if item["run_id"] == run_id
    )
    assert execution["status"] == expected_status.value
    reconciled = restarted.execute_replay(
        plan_id,
        run_id=run_id,
        candidate_key="ember",
        mode="strict_replay",
    )
    assert reconciled["status"] == expected_status.value
    assert reconciled["reconciled"] is True
    replay_stream = next(
        snapshot
        for snapshot in TypedEvidenceStore(tmp_path).iter_streams()
        if snapshot.stream_id.startswith("replay-")
        and any(event.payload.get("run_id") == run_id for event in snapshot.events)
    )
    assert [event.event_type for event in replay_stream.events] == [
        "replay_execution_started",
        terminal_event,
    ]
    assert prepared["plan_id"] == plan_id
