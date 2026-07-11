from __future__ import annotations

import importlib.util

import pytest


BASE = "0be94e0d67482d77e2e186f32086a705f09f4a88"
IDENTITY = {"type": "patch", "base": BASE, "patch": "sha256:synthetic-patch"}


def load_workflow_module():
    from pathlib import Path

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
    subject = {"type": "patch", "base": BASE, "patch": f"sha256:synthetic-{key}"}
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
        "review_triggers": [],
        "route": {"selected_model": "terra-high", "selected_effort": "high", "observed_model": "UNVERIFIED", "observed_effort": "UNVERIFIED"},
        "stop_conditions": ["scope drift"],
        "authorization_state": "MAINTAINER_AUTHORIZED",
        "classification": "protected_boundary_implementation",
        "mission_ref": "bounded workflow",
        "mission_fit": "FIT",
        "recursive_delegation": False,
        "subject": subject,
        "maintainer_authorization": {
            "actor": "maintainer",
            "task_key": key,
            "exact_base": BASE,
            "subject": subject,
            "task_packet_identity": subject,
            "authority_reference": f"authority:{key}",
            "readiness_evidence": "readiness:synthetic",
            "reviewed_corrections_evidence": "corrections:synthetic",
        },
    }


def verification_request():
    return {"command": ["python", "-m", "pytest"], "scope": ["tests/unit"], "environment": {"lock": "committed"}, "side_effects": []}


def final_gate_request():
    return {"command": ["python", "scripts/temper-gate.py", "all"], "scope": ["repository"], "environment": {"lock": "committed"}, "side_effects": []}


def verification_record(subject=IDENTITY, key="task-one", **extra):
    value = {"task_key": key, "reference": "verification:unit", "subject": subject, **verification_request(), "status": "PASS"}
    value.update(extra)
    return value


def decision(key="task-one", identity=IDENTITY):
    return {
        "kind": "decision", "task_key": key, "reference": "decision:task-one", "subject": identity,
        "authoritative_provenance": {"task_key": key, "subject": identity, "authority_reference": f"authority:{key}"},
    }


def handoff(key="task-one", worker_reference="writer-one", identity=IDENTITY, **extra):
    value = {
        "task_key": key,
        "worker_reference": worker_reference,
        "changed_paths": ["scripts/example.py"],
        "identity": identity,
        "route": {"selected_model": "terra-high", "selected_effort": "high"},
        "acceptance_evidence": ["focused test"],
        "verification": ["unit test"],
        "applied_decisions": [{"task_key": key, "reference": "decision:task-one", "identity": identity}],
        "verification_references": [{"task_key": key, "reference": "verification:unit", "identity": identity, "verification_request": verification_request()}],
        "scope_safety": "owned paths only",
        "open_findings": [],
        "integration_guidance": "maintainer decision required",
    }
    value.update(extra)
    return value


def to_plan(workflow, value, key="task-one"):
    result = workflow.advance(value, {"task_key": key})
    assert value["phase"] == "PLAN"
    assert result["next_action"] == "ACQUIRE_REQUIRED_OWNERSHIP"


def dispatch_worker(workflow, value, key="task-one"):
    to_plan(workflow, value, key)
    paths = value["tasks"][key]["owned_paths"]
    workflow.acquire_ownership(value, {"task_key": key, "paths": paths})
    workflow.advance(value, {"task_key": key})
    return workflow.register_worker(value, {"reference": "writer-one", "task_key": key})


def completed_candidate(workflow, value, key="task-one"):
    dispatch_worker(workflow, value, key)
    workflow.record_evidence(value, decision(key))
    workflow.ingest_handoff(value, handoff(key))
    workflow.record_evidence(value, {"kind": "verifier_registration", "task_key": key, "subject": IDENTITY, **final_gate_request(), "verifier_reference": "verifier:final", "accepted": True})
    workflow.record_verification(value, verification_record(key=key))
    workflow.record_verification(value, {"task_key": key, "reference": "verification:public", "subject": IDENTITY, "command": ["python", "scripts/public-safety.py"], "scope": ["repository"], "environment": {"lock": "committed"}, "side_effects": [], "status": "PASS", "verification_type": "public_safety"})
    workflow.record_verification(value, {"task_key": key, "reference": "verification:gate", "subject": IDENTITY, **final_gate_request(), "status": "PASS", "verifier_reference": "verifier:final"})


def test_canonical_cold_technical_review_class_is_routed():
    workflow = load_workflow_module()
    review = task(task_class="cold_technical_review")
    assert workflow.validate_task(review)["task_class"] == "cold_technical_review"


def test_authorization_phase_pairs_and_operations_fail_closed():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    with pytest.raises(workflow.WorkflowError, match="only valid in PLAN"):
        workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    to_plan(workflow, value)
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    with pytest.raises(workflow.WorkflowError, match="only valid for an implementing dispatch"):
        workflow.register_worker(value, {"reference": "writer-one", "task_key": "task-one"})
    with pytest.raises(workflow.WorkflowError, match="incompatible"):
        workflow.validate_state({**value, "phase": "VERIFY"})


def test_transition_cannot_walk_to_integration_or_grant_it_without_prerequisites():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.advance(value, {"task_key": "task-one"})
    workflow.register_worker(value, {"reference": "writer-one", "task_key": "task-one"})
    workflow.record_evidence(value, decision())
    workflow.ingest_handoff(value, handoff())
    with pytest.raises(workflow.WorkflowError, match="integration transition requires"):
        workflow.transition(value, "INTEGRATE")
    with pytest.raises(workflow.WorkflowError, match="verified candidate"):
        workflow.authorize(value, {"actor": "maintainer", "authorization_state": "INTEGRATION_AUTHORIZED"})


def test_path_collisions_include_ancestor_descendant_and_cross_platform_normalization():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task("parent", ["scripts"]))
    workflow.register_task(value, task("child", ["scripts\\demo.py"]))
    to_plan(workflow, value, "parent")
    workflow.acquire_ownership(value, {"task_key": "parent", "paths": ["scripts"]})
    value["authorization_state"] = "MAINTAINER_AUTHORIZED"
    with pytest.raises(workflow.WorkflowError, match="path ownership collision"):
        workflow.acquire_ownership(value, {"task_key": "child", "paths": ["scripts/demo.py"]})


def test_handoff_rejects_untyped_wrong_base_or_empty_subject_identities():
    workflow = load_workflow_module()
    for identity, message in (
        ({"base": BASE, "patch": "sha256:synthetic"}, "identity type"),
        ({"type": "patch", "base": "wrong-base", "patch": "sha256:synthetic"}, "identity base"),
        ({"type": "patch", "base": BASE, "patch": ""}, "non-empty patch"),
    ):
        value = state(workflow)
        workflow.register_task(value, task())
        dispatch_worker(workflow, value)
        workflow.record_evidence(value, decision())
        with pytest.raises(workflow.WorkflowError, match=message):
            workflow.ingest_handoff(value, handoff(identity=identity))


def test_handoff_requires_authoritative_structured_decisions_and_references():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    dispatch_worker(workflow, value)
    with pytest.raises(workflow.WorkflowError, match="authoritative task decision"):
        workflow.ingest_handoff(value, handoff())
    workflow.record_evidence(value, decision())
    invalid = handoff()
    invalid["verification_references"] = [{"task_key": "other", "reference": "verification:unit", "identity": IDENTITY}]
    with pytest.raises(workflow.WorkflowError, match="bind the handoff task"):
        workflow.ingest_handoff(value, invalid)


def test_verification_reuse_requires_typed_exact_subject_and_stable_inputs():
    workflow = load_workflow_module()
    complete = verification_record(untracked_inputs_relevant=True, untracked_inputs=[{"content_identity": "sha256:synthetic", "role": "fixture", "scope": "tests"}])
    assert workflow.verification_reuse(complete, IDENTITY, verification_request()) == {"reusable": True, "status": "REUSABLE"}
    assert workflow.verification_reuse(complete, {"type": "patch", "base": BASE, "patch": ""}, verification_request()) == {"reusable": False, "status": "NON_REUSABLE"}


def test_integration_requires_matching_handoff_bound_reusable_pass_and_decision():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    completed_candidate(workflow, value)
    assert workflow.advance(value, {"task_key": "task-one", "verification_request": verification_request()})["next_action"] == "AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION"
    request = {
        "actor": "maintainer",
        "authorization_state": "INTEGRATION_AUTHORIZED",
        "task_key": "task-one",
        "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0],
        "verification_reference": "verification:unit",
        "verification_request": verification_request(),
    }
    assert workflow.authorize(value, request) == {"authorization_state": "INTEGRATION_AUTHORIZED", "actor": "maintainer", "phase": "INTEGRATE"}


def test_integration_rejects_wrong_verification_identity_or_reference():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    completed_candidate(workflow, value)
    bad = {
        "actor": "maintainer", "authorization_state": "INTEGRATION_AUTHORIZED", "task_key": "task-one", "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0], "verification_reference": "other", "verification_request": verification_request(),
    }
    with pytest.raises(workflow.WorkflowError, match="handoff-bound verification"):
        workflow.authorize(value, bad)


def test_protected_review_still_allows_one_repair_cycle():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task(task_class="protected_boundary_implementation"))
    completed_candidate(workflow, value)
    record = {"task_key": "task-one", "verification_request": verification_request()}
    assert workflow.advance(value, record)["next_action"] == "AWAIT_COLD_REVIEW"
    result = workflow.advance(value, {**record, "review_status": "REPAIR_REQUIRED"})
    assert result["next_action"] == "REPAIR_CYCLE_PERMITTED"
    assert value["phase"] == "PLAN"
    assert value["authorization_state"] == "MAINTAINER_AUTHORIZED"


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


def test_state_rejects_orphan_active_writer_and_duplicate_task_key():
    workflow = load_workflow_module()
    value = state(workflow)
    value["workers"].append({"reference": "orphan", "task_key": "missing", "task_class": "normal_implementation", "writer": True, "status": "ACTIVE", "route": task()["route"]})
    with pytest.raises(workflow.WorkflowError, match="registered matching task"):
        workflow.validate_state(value)
    value = state(workflow)
    workflow.register_task(value, task())
    with pytest.raises(workflow.WorkflowError, match="already registered"):
        workflow.register_task(value, task())


def test_handoff_completes_worker_before_ownership_can_release():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    dispatch_worker(workflow, value)
    with pytest.raises(workflow.WorkflowError, match="while a task worker is active"):
        workflow.release_ownership(value, {"task_key": "task-one"})
    workflow.record_evidence(value, decision())
    workflow.ingest_handoff(value, handoff())
    assert workflow.release_ownership(value, {"task_key": "task-one"}) == {"task_key": "task-one", "released": True}


def test_handoff_rejects_empty_evidence_and_route_mismatch():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    dispatch_worker(workflow, value)
    workflow.record_evidence(value, decision())
    incomplete = handoff()
    incomplete["acceptance_evidence"] = []
    with pytest.raises(workflow.WorkflowError, match="acceptance_evidence"):
        workflow.ingest_handoff(value, incomplete)
    wrong_route = handoff()
    wrong_route["route"]["selected_model"] = "terra-medium"
    with pytest.raises(workflow.WorkflowError, match="route"):
        workflow.ingest_handoff(value, wrong_route)


def test_context_is_typed_and_does_not_replay_evidence_transcripts():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    value["evidence"].append({"kind": "note", "subject": {"id": "synthetic"}, "transcript": "do not copy"})
    context = workflow.compile_context(value)
    assert context["mission"]["priority"] == "bounded workflow"
    assert "evidence" not in context
    assert "transcript" not in str(context)


def test_verification_binding_rejects_cross_task_reference_reuse():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    workflow.register_task(value, task("task-two", ["tests/example.py"]))
    completed_candidate(workflow, value)
    workflow.record_verification(value, {**verification_record(key="task-two"), "reference": "verification:task-two"})
    request = {
        "actor": "maintainer", "authorization_state": "INTEGRATION_AUTHORIZED", "task_key": "task-one", "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0], "verification_reference": "verification:task-two", "verification_request": verification_request(),
    }
    with pytest.raises(workflow.WorkflowError, match="handoff-bound verification"):
        workflow.authorize(value, request)


def test_integration_requires_cold_review_final_gate_and_public_safety_evidence():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task(task_class="protected_boundary_implementation"))
    completed_candidate(workflow, value)
    record = {"task_key": "task-one", "verification_request": verification_request()}
    workflow.advance(value, record)
    workflow.advance(value, {**record, "review_status": "PASS"})
    request = {
        "actor": "maintainer", "authorization_state": "INTEGRATION_AUTHORIZED", "task_key": "task-one", "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0], "verification_reference": "verification:unit", "verification_request": verification_request(),
    }
    with pytest.raises(workflow.WorkflowError, match="registered-verifier"):
        workflow.authorize(value, request)
    workflow.record_verification(value, {"task_key": "task-one", "reference": "verification:gate-after-review", "subject": IDENTITY, **final_gate_request(), "status": "PASS", "verifier_reference": "verifier:final"})
    assert workflow.authorize(value, request)["phase"] == "INTEGRATE"


def test_registered_risk_trigger_cannot_be_omitted_from_advance():
    workflow = load_workflow_module()
    value = state(workflow)
    record = task()
    record["review_triggers"] = ["public_repository_safety"]
    workflow.register_task(value, record)
    completed_candidate(workflow, value)
    result = workflow.advance(value, {"task_key": "task-one", "verification_request": verification_request()})
    assert result["next_action"] == "AWAIT_COLD_REVIEW"
    assert result["review_packet"]["subject"] == IDENTITY


def test_final_and_public_safety_evidence_use_registered_verifier_and_latest_result():
    workflow = load_workflow_module()
    request = {
        "actor": "maintainer", "authorization_state": "INTEGRATION_AUTHORIZED", "task_key": "task-one", "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0], "verification_reference": "verification:unit", "verification_request": verification_request(),
    }
    arbitrary = state(workflow)
    workflow.register_task(arbitrary, task())
    completed_candidate(workflow, arbitrary)
    workflow.record_verification(arbitrary, {"task_key": "task-one", "reference": "verification:arbitrary-gate", "subject": IDENTITY, **final_gate_request(), "status": "PASS", "verifier_reference": "unregistered:label"})
    with pytest.raises(workflow.WorkflowError, match="registered-verifier"):
        workflow.authorize(arbitrary, request)

    stale_gate = state(workflow)
    workflow.register_task(stale_gate, task())
    completed_candidate(workflow, stale_gate)
    workflow.record_verification(stale_gate, {"task_key": "task-one", "reference": "verification:gate", "subject": IDENTITY, **final_gate_request(), "status": "FAIL"})
    assert stale_gate["authorization_state"] == "VERIFIED"
    with pytest.raises(workflow.WorkflowError, match="registered-verifier"):
        workflow.authorize(stale_gate, request)

    stale_safety = state(workflow)
    workflow.register_task(stale_safety, task())
    completed_candidate(workflow, stale_safety)
    workflow.record_verification(stale_safety, {"task_key": "task-one", "reference": "verification:public", "subject": IDENTITY, "command": ["python", "scripts/public-safety.py"], "scope": ["repository"], "environment": {"lock": "committed"}, "side_effects": [], "status": "FAIL"})
    with pytest.raises(workflow.WorkflowError, match="public-safety"):
        workflow.authorize(stale_safety, request)


def test_authority_provenance_and_observed_route_mismatch_fail_closed():
    workflow = load_workflow_module()
    invalid = task()
    del invalid["maintainer_authorization"]["readiness_evidence"]
    with pytest.raises(workflow.WorkflowError, match="provenance"):
        workflow.validate_task(invalid, exact_base=BASE)
    invalid = task()
    invalid["route"]["observed_model"] = "other-model"
    with pytest.raises(workflow.WorkflowError, match="observed_model"):
        workflow.validate_task(invalid, exact_base=BASE)


def test_decision_authority_must_match_registered_task_authority():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    unrelated = decision()
    unrelated["authoritative_provenance"]["authority_reference"] = "authority:unrelated"
    with pytest.raises(workflow.WorkflowError, match="authoritative"):
        workflow.record_evidence(value, unrelated)
    workflow.record_evidence(value, decision())
    value["evidence"][-1]["authoritative_provenance"]["authority_reference"] = "authority:unrelated"
    with pytest.raises(workflow.WorkflowError, match="authoritative"):
        workflow.validate_state(value)


def test_read_only_tasks_and_windows_absolute_paths_cannot_be_written():
    workflow = load_workflow_module()
    for path in ("C:\\work\\file.py", "C:work\\file.py", "\\rooted\\file.py", "\\\\server\\share\\file.py", "\\\\?\\C:\\file.py", "/repo/file.py"):
        with pytest.raises(workflow.WorkflowError, match="unsafe owned path"):
            workflow.normalized_path(path)
    value = state(workflow)
    workflow.register_task(value, task(task_class="cold_technical_review"))
    to_plan(workflow, value)
    with pytest.raises(workflow.WorkflowError, match="read-only"):
        workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})


def test_routine_tasks_are_read_only_and_worker_flags_or_routes_cannot_override_task_policy():
    workflow = load_workflow_module()
    routine = task(task_class="routine_administration")
    routine["route"] = {"selected_model": "luna_or_cheapest", "selected_effort": "low", "observed_model": "UNVERIFIED", "observed_effort": "UNVERIFIED"}
    value = state(workflow)
    workflow.register_task(value, routine)
    to_plan(workflow, value)
    with pytest.raises(workflow.WorkflowError, match="read-only"):
        workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})

    candidate = state(workflow)
    implementation = task()
    implementation["route"]["observed_model"] = "terra-high"
    implementation["route"]["observed_effort"] = "high"
    workflow.register_task(candidate, implementation)
    to_plan(workflow, candidate)
    workflow.acquire_ownership(candidate, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.advance(candidate, {"task_key": "task-one"})
    with pytest.raises(workflow.WorkflowError, match="derived"):
        workflow.register_worker(candidate, {"reference": "writer-one", "task_key": "task-one", "writer": False})
    mismatched = task()["route"]
    with pytest.raises(workflow.WorkflowError, match="supplied route"):
        workflow.register_worker(candidate, {"reference": "writer-one", "task_key": "task-one", "route": mismatched})


def test_guarded_second_writer_is_reachable_and_third_or_unguarded_writer_is_rejected():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task("task-one", ["scripts/one.py"]))
    workflow.register_task(value, task("task-two", ["scripts/two.py"]))
    workflow.register_task(value, task("task-three", ["scripts/three.py"]))
    to_plan(workflow, value)
    for key in ("task-one", "task-two", "task-three"):
        workflow.acquire_ownership(value, {"task_key": key, "paths": value["tasks"][key]["owned_paths"]})
    workflow.advance(value, {"task_key": "task-one"})
    workflow.register_worker(value, {"reference": "writer-one", "task_key": "task-one"})
    with pytest.raises(workflow.WorkflowError, match="durable task-specific"):
        workflow.register_worker(value, {"reference": "writer-two", "task_key": "task-two"})
    with pytest.raises(workflow.WorkflowError, match="independence guard"):
        workflow.advance(value, {"task_key": "task-two"})
    guards = {name: True for name in workflow.REQUIRED_SECOND_WRITER_GUARDS}
    assert workflow.advance(value, {"task_key": "task-two", "independence_guards": guards})["advanced"] is True
    assert workflow.register_worker(value, {"reference": "writer-two", "task_key": "task-two"})["status"] == "ACTIVE"
    with pytest.raises(workflow.WorkflowError, match="third"):
        workflow.advance(value, {"task_key": "task-three", "independence_guards": guards})


def test_duplicate_task_attempt_and_ambiguous_spawn_reconciliation_are_rejected_or_durable():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.advance(value, {"task_key": "task-one"})
    value["workers"].append({"reference": "duplicate", "task_key": "task-one", "task_class": "normal_implementation", "writer": True, "status": "ACTIVE", "route": task()["route"], "attempt": 2, "monitor": workflow._new_monitor("ACTIVE")})
    with pytest.raises(workflow.WorkflowError, match="for a task"):
        workflow.validate_state(value)
    value["workers"].pop()
    assert workflow.reconcile_worker(value, {"reference": "manual:task-one", "status": "SPAWN_UNKNOWN", "ambiguity_reference": "ambiguity:synthetic"})["status"] == "SPAWN_UNKNOWN"
    assert workflow.reconcile_worker(value, {"reference": "manual:task-one"})["retry"] == "BLOCKED"


def test_newer_fail_invalidates_only_its_task_verification_pass():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    workflow.register_task(value, task("task-two", ["tests/example.py"]))
    completed_candidate(workflow, value)
    workflow.record_verification(value, {**verification_record(), "status": "FAIL"})
    assert value["authorization_state"] == "IMPLEMENTING"
    assert workflow.advance(value, {"task_key": "task-one", "verification_request": verification_request()})["next_action"] == "RUN_OR_RECORD_REQUIRED_VERIFICATION"
    different_request = state(workflow)
    workflow.register_task(different_request, task())
    completed_candidate(workflow, different_request)
    workflow.record_verification(different_request, {"task_key": "task-one", "reference": "verification:other", "subject": IDENTITY, "command": ["python", "-m", "pytest", "tests/other"], "scope": ["tests/other"], "environment": {"lock": "committed"}, "side_effects": [], "status": "FAIL"})
    assert different_request["authorization_state"] == "VERIFIED"
    assert workflow._reusable_verification(different_request, "task-one", IDENTITY, verification_request())["status"] == "PASS"
    separate = state(workflow)
    workflow.register_task(separate, task())
    workflow.register_task(separate, task("task-two", ["tests/example.py"]))
    completed_candidate(workflow, separate)
    workflow.record_verification(separate, {**verification_record(key="task-two"), "reference": "verification:task-two", "status": "FAIL"})
    assert separate["authorization_state"] == "VERIFIED"
    assert workflow._reusable_verification(separate, "task-one", IDENTITY, verification_request())["status"] == "PASS"


def test_worker_heartbeat_backs_off_without_replaying_context_or_spawning_duplicates():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.advance(value, {"task_key": "task-one"})
    first = workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 100})
    assert first == {"action": "WAIT_UNTIL_HEARTBEAT", "reference": "manual:task-one", "interval_seconds": 120, "next_check_at": 220}
    early = workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 150})
    assert early == first
    backed_off = workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 220})
    assert backed_off["action"] == "WAIT_UNTIL_HEARTBEAT"
    assert backed_off["interval_seconds"] == 240
    assert len(value["workers"]) == 1
    assert workflow.advance(value, {"task_key": "task-one"})["next_action"] == "AWAIT_MANUAL_WORKER_REFERENCE"
    assert len(value["workers"]) == 1


def test_worker_heartbeat_resets_on_change_and_wakes_immediately_for_terminal_events():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    dispatch_worker(workflow, value)
    workflow.monitor_worker(value, {"reference": "writer-one", "now": 0})
    workflow.monitor_worker(value, {"reference": "writer-one", "now": 120})
    changed = workflow.monitor_worker(value, {"reference": "writer-one", "now": 121, "output_identity": "output:synthetic"})
    assert changed["action"] == "HEARTBEAT_CHANGED"
    assert changed["interval_seconds"] == 120
    assert workflow.monitor_worker(value, {"reference": "writer-one", "now": 122, "system_error": True})["reason"] == "SYSTEM_ERROR"
    assert workflow.monitor_worker(value, {"reference": "writer-one", "now": 123, "maintainer_input": True})["reason"] == "MAINTAINER_INPUT"
    completed = workflow.monitor_worker(value, {"reference": "writer-one", "now": 124, "status": "COMPLETE"})
    assert completed == {"action": "WAKE_IMMEDIATELY", "reason": "COMPLETION", "reference": "writer-one"}


def test_worker_heartbeat_requires_durable_ambiguous_spawn_reconciliation():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(value, {"task_key": "task-one", "paths": ["scripts/example.py"]})
    workflow.advance(value, {"task_key": "task-one"})
    with pytest.raises(workflow.WorkflowError, match="ambiguity_reference"):
        workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 0, "status": "SPAWN_UNKNOWN"})
    immediate = workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 0, "status": "SPAWN_UNKNOWN", "ambiguity_reference": "ambiguity:synthetic"})
    assert immediate["reason"] == "AMBIGUOUS_SPAWN_RECONCILIATION"
    assert workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 1})["reason"] == "AMBIGUOUS_SPAWN_RECONCILIATION"
