from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


BASE = "0be94e0d67482d77e2e186f32086a705f09f4a88"


def load_workflow_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "temper-workflow.py"
    spec = importlib.util.spec_from_file_location("temper_workflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def state(workflow):
    value = workflow.default_state()
    value["authorization_state"] = "MAINTAINER_AUTHORIZED"
    value["mission"] = {"priority": "bounded workflow", "mission_fit": "FIT"}
    value["repository"] = {"exact_base": BASE}
    return value


def task(key="task-one", paths=None, task_class="normal_implementation"):
    return {
        "task_key": key,
        "task_class": task_class,
        "objective": "Implement only the bounded coordinator record.",
        "exact_base": BASE,
        "owned_paths": paths or ["scripts/example.py"],
        "acceptance_criteria": ["focused test"],
        "non_goals": ["product change"],
        "verification": ["unit test"],
        "review": ["matrix"],
        "route": {
            "selected_model": "terra-high",
            "selected_effort": "high",
            "observed_model": "UNVERIFIED",
            "observed_effort": "UNVERIFIED",
        },
        "stop_conditions": ["scope drift"],
        "authorization_state": "MAINTAINER_AUTHORIZED",
        "classification": "protected_boundary_implementation",
        "mission_ref": "bounded workflow",
        "mission_fit": "FIT",
        "recursive_delegation": False,
    }


def verification_record(subject, **extra):
    value = {
        "subject": subject,
        "command": ["python", "-m", "pytest"],
        "scope": ["tests/unit"],
        "environment": {"lock": "committed"},
        "side_effects": [],
        "status": "PASS",
    }
    value.update(extra)
    return value


def verification_request():
    return {
        "command": ["python", "-m", "pytest"],
        "scope": ["tests/unit"],
        "environment": {"lock": "committed"},
        "side_effects": [],
    }


def handoff(key="task-one", worker_reference="writer-one", changed_paths=None):
    return {
        "task_key": key,
        "worker_reference": worker_reference,
        "changed_paths": changed_paths or ["scripts/example.py"],
        "identity": {"base": BASE, "patch": "synthetic-patch"},
        "route": {"selected_model": "terra-high", "selected_effort": "high"},
        "acceptance_evidence": ["focused test"],
        "verification": ["unit test"],
        "scope_safety": "owned paths only",
        "open_findings": [],
        "integration_guidance": "maintainer decision required",
    }


def test_task_validation_rejects_recursive_delegation_and_unknown_selection():
    workflow = load_workflow_module()
    record = task()
    record["recursive_delegation"] = True
    with pytest.raises(workflow.WorkflowError, match="recursive"):
        workflow.validate_task(record, exact_base=BASE)

    record = task()
    record["route"]["selected_model"] = ""
    with pytest.raises(workflow.WorkflowError, match="selected model"):
        workflow.validate_task(record, exact_base=BASE)

    record = task()
    record["task_class"] = "unsupported"
    with pytest.raises(workflow.WorkflowError, match="unsupported task class"):
        workflow.validate_task(record, exact_base=BASE)


def test_invalid_transition_and_invalid_checkpoint_state_fail_closed():
    workflow = load_workflow_module()
    value = state(workflow)
    with pytest.raises(workflow.WorkflowError, match="invalid transition"):
        workflow.transition(value, "VERIFY")

    value["phase"] = "NOT_A_PHASE"
    with pytest.raises(workflow.WorkflowError, match="unknown workflow phase"):
        workflow.checkpoint(value, {"next_action": "stop"})


def test_duplicate_task_key_and_exact_path_ownership_are_rejected():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    with pytest.raises(workflow.WorkflowError, match="already registered"):
        workflow.register_task(value, task())

    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.register_task(value, task("task-two", ["scripts/example.py"]))
    with pytest.raises(workflow.WorkflowError, match="collision"):
        workflow.acquire_ownership(value, {"task_key": "task-two", "paths": ["scripts/example.py"]})


def test_acquire_dispatch_and_worker_registration_require_current_authorization_and_lease():
    workflow = load_workflow_module()

    value = state(workflow)
    workflow.register_task(value, task())
    value["authorization_state"] = "PROPOSED"
    with pytest.raises(workflow.WorkflowError, match="currently maintainer-authorized"):
        workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})

    value = state(workflow)
    workflow.register_task(value, task())
    value["phase"] = "PLAN"
    with pytest.raises(workflow.WorkflowError, match="active ownership lease"):
        workflow.advance(value, {"task_key": "task-one"})

    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    value["authorization_state"] = "PROPOSED"
    assert workflow.advance(value, {"task_key": "task-one"})["next_action"] == "STOP_FOR_MAINTAINER_AUTHORIZATION"
    with pytest.raises(workflow.WorkflowError, match="currently maintainer-authorized"):
        workflow.register_worker(value, {"reference": "writer-one", "task_key": "task-one"})


def test_spawn_unknown_blocks_retry_until_maintainer_replacement():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    value["workers"].append(
        {
            "reference": "manual:task-one",
            "task_key": "task-one",
            "task_class": "normal_implementation",
            "writer": True,
            "status": "SPAWN_UNKNOWN",
            "route": task()["route"],
        }
    )

    assert workflow.reconcile_worker(value, {"reference": "manual:task-one"})["retry"] == "BLOCKED"
    with pytest.raises(workflow.WorkflowError, match="SPAWN_UNKNOWN"):
        workflow.register_worker(value, {"reference": "worker-two", "task_key": "task-one"})
    assert workflow.reconcile_worker(value, {"reference": "manual:task-one", "maintainer_replacement_decision": True})["status"] == "RETIRED"


def test_corrupt_active_writer_records_fail_closed():
    workflow = load_workflow_module()
    value = state(workflow)
    value["workers"].append(
        {
            "reference": "orphan",
            "task_key": "missing-task",
            "task_class": "normal_implementation",
            "writer": True,
            "status": "ACTIVE",
            "route": task()["route"],
        }
    )
    with pytest.raises(workflow.WorkflowError, match="registered matching task"):
        workflow.validate_state(value)

    value = state(workflow)
    workflow.register_task(value, task())
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    value["workers"].append(
        {
            "reference": "wrong-class",
            "task_key": "task-one",
            "task_class": "technical_review",
            "writer": True,
            "status": "ACTIVE",
            "route": task()["route"],
        }
    )
    with pytest.raises(workflow.WorkflowError, match="task class"):
        workflow.validate_state(value)


def test_one_writer_default_two_guarded_and_three_rejected():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task("task-one", ["scripts/one.py"]))
    workflow.register_task(value, task("task-two", ["scripts/two.py"]))
    workflow.register_task(value, task("task-three", ["scripts/three.py"]))
    for key, path in (("task-one", "scripts/one.py"), ("task-two", "scripts/two.py"), ("task-three", "scripts/three.py")):
        workflow.acquire_ownership(value, {"task_key": key, "paths": [path]})

    workflow.register_worker(value, {"reference": "writer-one", "task_key": "task-one"})
    with pytest.raises(workflow.WorkflowError, match="second writer"):
        workflow.register_worker(value, {"reference": "writer-two", "task_key": "task-two"})
    workflow.register_worker(
        value,
        {
            "reference": "writer-two",
            "task_key": "task-two",
            "independence_guards": {name: True for name in workflow.REQUIRED_SECOND_WRITER_GUARDS},
        },
    )
    with pytest.raises(workflow.WorkflowError, match="third"):
        workflow.register_worker(
            value,
            {
                "reference": "writer-three",
                "task_key": "task-three",
                "independence_guards": {name: True for name in workflow.REQUIRED_SECOND_WRITER_GUARDS},
            },
        )


def test_verification_reuse_requires_untracked_input_identities():
    workflow = load_workflow_module()
    subject = {"base": BASE, "patch": "synthetic-patch"}
    missing = verification_record(subject, untracked_inputs_relevant=True, untracked_inputs=[{"role": "fixture", "scope": "tests"}])
    assert workflow.verification_reuse(missing, subject, verification_request()) == {"reusable": False, "status": "NON_REUSABLE"}
    complete = verification_record(subject, untracked_inputs_relevant=True, untracked_inputs=[{"content_identity": "sha256:synthetic", "role": "fixture", "scope": "tests"}])
    assert workflow.verification_reuse(complete, subject, verification_request()) == {"reusable": True, "status": "REUSABLE"}
    different_command = verification_request()
    different_command["command"] = ["python", "-m", "compileall"]
    assert workflow.verification_reuse(complete, subject, different_command) == {"reusable": False, "status": "NON_REUSABLE"}
    different_side_effects = verification_request()
    different_side_effects["side_effects"] = ["generated cache"]
    assert workflow.verification_reuse(complete, subject, different_side_effects) == {"reusable": False, "status": "NON_REUSABLE"}


def test_context_is_typed_and_does_not_replay_evidence_transcripts():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    value["evidence"].append({"kind": "decision", "subject": {"id": "synthetic"}, "transcript": "do not copy"})
    context = workflow.compile_context(value)
    assert context["mission"]["priority"] == "bounded workflow"
    assert "evidence" not in context
    assert "transcript" not in json.dumps(context)


def test_review_is_zero_or_one_and_protected_boundary_is_cold_terra_high():
    workflow = load_workflow_module()
    protected = task(task_class="protected_boundary_implementation")
    result = workflow.review_required(protected, ["public_repository_safety", "weak_evidence"])
    assert result == {"required": True, "reviewers": 1, "route": "terra-high", "reason": "protected_boundary"}
    assert workflow.review_required(task(), []) == {"required": False, "reviewers": 0, "route": None, "reason": "no_matrix_trigger"}


def test_advance_emits_manual_packet_and_does_not_launch_a_worker():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    value["phase"] = "PLAN"

    result = workflow.advance(value, {"task_key": "task-one"})
    assert result["manual_launch_packet"]["manual_adapter"] is True
    assert result["next_action"] == "AWAIT_MANUAL_WORKER_REFERENCE"
    assert value["workers"][0]["status"] == "SPAWN_REQUESTED"
    assert value["workers"][0]["reference"] == "manual:task-one"
    registered = workflow.register_worker(value, {"reference": "manual-worker", "task_key": "task-one"})
    assert registered["status"] == "ACTIVE"
    assert len(value["workers"]) == 1
    with pytest.raises(workflow.WorkflowError, match="already has an active worker"):
        workflow.register_worker(
            value,
            {
                "reference": "another-worker",
                "task_key": "task-one",
                "independence_guards": {name: True for name in workflow.REQUIRED_SECOND_WRITER_GUARDS},
            },
        )


def test_handoff_completes_exact_worker_before_ownership_can_release():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.register_worker(value, {"reference": "writer-one", "task_key": "task-one"})

    with pytest.raises(workflow.WorkflowError, match="while a task worker is active"):
        workflow.release_ownership(value, {"task_key": "task-one"})
    with pytest.raises(workflow.WorkflowError, match="does not belong"):
        workflow.ingest_handoff(value, handoff(worker_reference="other-worker"))

    result = workflow.ingest_handoff(value, handoff())
    assert result == {"task_key": "task-one", "worker_reference": "writer-one", "ingested": True}
    assert value["workers"][0]["status"] == "COMPLETE"
    assert workflow.release_ownership(value, {"task_key": "task-one"}) == {"task_key": "task-one", "released": True}


def test_handoff_rejects_spawn_states_empty_evidence_and_route_mismatch():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    value["workers"].append(
        {
            "reference": "manual:task-one",
            "task_key": "task-one",
            "task_class": "normal_implementation",
            "writer": True,
            "status": "SPAWN_REQUESTED",
            "route": task()["route"],
        }
    )
    with pytest.raises(workflow.WorkflowError, match="not active"):
        workflow.ingest_handoff(value, handoff(worker_reference="manual:task-one"))

    value["workers"][0]["status"] = "ACTIVE"
    missing_evidence = handoff(worker_reference="manual:task-one")
    missing_evidence["acceptance_evidence"] = []
    with pytest.raises(workflow.WorkflowError, match="acceptance_evidence"):
        workflow.ingest_handoff(value, missing_evidence)
    wrong_route = handoff(worker_reference="manual:task-one")
    wrong_route["route"]["selected_model"] = "terra-medium"
    with pytest.raises(workflow.WorkflowError, match="route"):
        workflow.ingest_handoff(value, wrong_route)


def test_advance_auto_progresses_then_stops_for_review_repair_and_integration():
    workflow = load_workflow_module()
    value = state(workflow)
    normal = task(task_class="normal_implementation")
    workflow.register_task(value, normal)
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})

    result = workflow.advance(value, {"task_key": "task-one"})
    assert result["automatic_transitions"] == ["DELIBERATE", "DECIDE", "PLAN", "DISPATCH"]
    workflow.register_worker(value, {"reference": "normal-worker", "task_key": "task-one"})
    workflow.ingest_handoff(value, handoff(worker_reference="normal-worker"))
    assert value["phase"] == "VERIFY"
    request = verification_request()
    assert workflow.advance(value, {"task_key": "task-one", "verification_request": request})["next_action"] == "RUN_OR_RECORD_REQUIRED_VERIFICATION"
    workflow.record_verification(value, verification_record(handoff()["identity"]))
    result = workflow.advance(value, {"task_key": "task-one", "verification_request": request})
    assert result["automatic_transitions"] == ["INTEGRATE"]
    assert result["integration_evidence"]["review"] == "NOT_REQUIRED"
    assert result["integration_evidence"]["verification"]["status"] == "PASS"
    assert workflow.advance(value, {})["next_action"] == "AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION"

    value = state(workflow)
    workflow.register_task(value, task(task_class="protected_boundary_implementation"))
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.register_worker(value, {"reference": "review-worker", "task_key": "task-one"})
    value["phase"] = "DISPATCH"
    workflow.ingest_handoff(value, handoff(worker_reference="review-worker"))
    workflow.record_verification(value, verification_record(handoff()["identity"]))
    review_record = {"task_key": "task-one", "verification_request": verification_request()}
    review_wait = workflow.advance(value, review_record)
    assert review_wait["next_action"] == "AWAIT_COLD_REVIEW"
    assert review_wait["review_packet"]["route"] == "terra-high"
    assert review_wait["review_packet"]["write_ownership"] == "none"
    repair = workflow.advance(value, {**review_record, "review_status": "REPAIR_REQUIRED"})
    assert repair["next_action"] == "REPAIR_CYCLE_PERMITTED"
    assert value["phase"] == "PLAN"

    value["phase"] = "VERIFY"
    limit = workflow.advance(value, {**review_record, "review_status": "REPAIR_REQUIRED"})
    assert limit["next_action"] == "REPAIR_LIMIT_REACHED"
    assert workflow.advance(value, {**review_record, "review_status": "PASS"})["next_action"] == "REPAIR_LIMIT_REACHED"


def test_cli_advance_persists_phase_neutral_review_mutations(tmp_path):
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task(task_class="protected_boundary_implementation"))
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.register_worker(value, {"reference": "review-worker", "task_key": "task-one"})
    value["phase"] = "DISPATCH"
    workflow.ingest_handoff(value, handoff(worker_reference="review-worker"))
    workflow.record_verification(value, verification_record(handoff()["identity"]))
    path = tmp_path / "review-state.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    record = json.dumps({"task_key": "task-one", "verification_request": verification_request()})

    assert workflow.main(["--state", str(path), "--record", record, "advance"]) == 0
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "VERIFY"
    assert persisted["reviews"][0]["status"] == "REQUIRED"

    persisted["reviews"][0]["status"] = "REPAIR_IN_PROGRESS"
    persisted["reviews"][0]["repair_cycles"] = 1
    path.write_text(json.dumps(persisted), encoding="utf-8")
    limit_record = json.dumps({
        "task_key": "task-one",
        "verification_request": verification_request(),
        "review_status": "REPAIR_REQUIRED",
    })
    assert workflow.main(["--state", str(path), "--record", limit_record, "advance"]) == 0
    limited = json.loads(path.read_text(encoding="utf-8"))
    assert limited["phase"] == "VERIFY"
    assert limited["reviews"][0]["status"] == "REPAIR_LIMIT_REACHED"


def test_cli_writes_only_mutating_commands(tmp_path, capsys):
    workflow = load_workflow_module()
    record = state(workflow)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    assert workflow.main(["--state", str(path), "status"]) == 0
    before = path.read_text(encoding="utf-8")
    assert workflow.main(["--state", str(path), "authorize", "--record", '{"actor":"maintainer"}']) == 0
    assert path.read_text(encoding="utf-8") != before
    assert "MAINTAINER_AUTHORIZED" in capsys.readouterr().out
