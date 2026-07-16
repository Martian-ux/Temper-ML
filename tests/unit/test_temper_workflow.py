from __future__ import annotations

import copy
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


def route(task_class="normal_implementation"):
    if task_class == "routine_administration":
        declared_route = "luna-or-cheapest"
        model = "gpt-5.6-luna"
        effort = "low"
    elif task_class == "mechanical_change":
        declared_route = "terra-medium"
        model = "gpt-5.6-terra"
        effort = "medium"
    else:
        declared_route = "terra-high"
        model = "gpt-5.6-terra"
        effort = "high"
    return {
        "declared_route": declared_route,
        "declared_model": model,
        "declared_effort": effort,
        "selected_model": model,
        "selected_effort": effort,
        "selection_mechanism": {
            "kind": "host_controls",
            "model_selector": "model",
            "effort_selector": "reasoning_effort",
            "control_surface": "synthetic_host_controls",
        },
        "runtime_observation": {
            "availability": "UNAVAILABLE",
            "source": "synthetic_host_without_telemetry",
            "model": "UNVERIFIED",
            "effort": "UNVERIFIED",
        },
        "declared_route_compliance": "UNVERIFIED",
        "experiment": {"label": "NOT_EXPERIMENT", "predeclared": False},
    }


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
        "route": route(task_class),
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
    return {
        "command": ["python", "-m", "pytest"],
        "scope": ["tests/unit"],
        "environment": {"lock": "committed"},
        "side_effects": [],
    }


def final_gate_request():
    return {
        "command": ["python", "scripts/temper-gate.py", "all"],
        "scope": ["repository"],
        "environment": {"lock": "committed"},
        "side_effects": [],
    }


def verification_record(subject=IDENTITY, key="task-one", **extra):
    value = {
        "task_key": key,
        "reference": "verification:unit",
        "subject": subject,
        **verification_request(),
        "status": "PASS",
    }
    value.update(extra)
    return value


def decision(key="task-one", identity=IDENTITY):
    return {
        "kind": "decision",
        "task_key": key,
        "reference": "decision:task-one",
        "subject": identity,
        "authoritative_provenance": {
            "task_key": key,
            "subject": identity,
            "authority_reference": f"authority:{key}",
        },
    }


def matched_experiment(label, pair_id, run_id, task_mix):
    return {
        "label": label,
        "predeclared": True,
        "trial_id": "trial-synthetic",
        "pair_id": pair_id,
        "run_id": run_id,
        "protocol_ref": "docs/workflow/procedures/model-route-and-experiment.md",
        "frozen_task_identity": f"sha256:frozen-{pair_id}",
        "isolation_ref": f"isolation:{pair_id}",
        "task_mix": task_mix,
    }


def trial_score(quality, ledger_ref):
    return {
        "ledger_ref": ledger_ref,
        "raw": {
            "input_tokens": 100,
            "cached_input_tokens": 20,
            "output_tokens": 20,
            "effective_tokens": 100,
            "elapsed_seconds": 10,
            "agent_sessions": 1,
            "redundant_full_gate_runs": 0,
            "credited_ledger_weight": quality // 10,
            "total_ledger_weight": 10,
            "false_positives": 0,
            "acceptance_complete": True,
            "avoidable_blocking_clarifications": 0,
            "avoidable_tool_or_test_retries": 0,
            "incomplete_outcome": 0,
            "handoff_passed_items": 10,
        },
        "components": {
            "quality": quality,
            "autonomy": 100,
            "efficiency": 100,
            "handoff": 100,
            "total": 0.50 * quality + 50,
        },
        "ledger_evidence_refs": ["evidence:synthetic"],
    }


def matched_trial():
    task_mix = [
        "fixed_snapshot_cold_review",
        "fixed_snapshot_cold_review",
        "bounded_implementation_or_repair",
        "bounded_implementation_or_repair",
        "architecture_or_invariant_design",
        "mechanical_negative_control",
    ]
    pairs = []
    for index, mix in enumerate(task_mix, start=1):
        pair_id = f"pair-{index}"
        runs = []
        for label, quality in (("CONTROL", 80), ("EXPERIMENTAL", 90)):
            run_id = f"run-{index}-{label.casefold()}"
            selected_route = route()
            if label == "EXPERIMENTAL":
                selected_route.update(
                    {
                        "declared_route": "sol-ultra",
                        "declared_model": "gpt-5.6-sol",
                        "declared_effort": "ultra",
                        "selected_model": "gpt-5.6-sol",
                        "selected_effort": "ultra",
                    }
                )
            selected_route["experiment"] = matched_experiment(
                label, pair_id, run_id, mix
            )
            runs.append(
                {
                    "run_id": run_id,
                    "context_identity": f"context-{index}-{label.casefold()}",
                    "task_class": "normal_implementation",
                    "route": selected_route,
                    "score": trial_score(quality, f"ledger:{pair_id}"),
                }
            )
        pairs.append(
            {
                "pair_id": pair_id,
                "task_mix": mix,
                "exact_base": BASE,
                "frozen_task_identity": f"sha256:frozen-{pair_id}",
                "isolation_ref": f"isolation:{pair_id}",
                "isolation_verified": True,
                "ledger_ref": f"ledger:{pair_id}",
                "runs": runs,
            }
        )
    return {
        "trial_id": "trial-synthetic",
        "classification": "MATCHED_TRIAL",
        "status": "COMPLETE",
        "protocol_ref": "docs/workflow/procedures/model-route-and-experiment.md",
        "eligible_for_default_route_decision": True,
        "adjudication": {
            "blinded": True,
            "alias_seed": "seed:synthetic",
            "adjudicator_ref": "adjudicator:synthetic",
            "ledger_ref": "ledger:synthetic",
            "formula_ref": "docs/workflow/procedures/model-route-and-experiment.md",
        },
        "pairs": pairs,
        "aggregate": {
            "report_ref": "report:synthetic",
            "p1_p2_differences_ref": "findings:synthetic",
            "mean_total_delta": 5,
            "median_total_delta": 5,
            "mean_quality_delta": 10,
            "escaped_p1": False,
            "outcome": "MATERIAL_QUALITY_BENEFIT",
            "recommendation": "ADOPT_EXPERIMENTAL",
        },
    }


def writer_exception_decision(value, key="task-one", writer_mode="subagent"):
    registered = value["tasks"][key]
    identity = registered["subject"]
    reference = f"writer-exception:{key}:{writer_mode}"
    return {
        "kind": "decision",
        "decision_type": "exceptional_writer",
        "actor": "maintainer",
        "approved": True,
        "task_key": key,
        "reference": reference,
        "subject": identity,
        "exception_scope": {
            "task_key": key,
            "subject": identity,
            "exact_base": registered["exact_base"],
            "owned_paths": registered["owned_paths"],
            "writer_mode": writer_mode,
            "reason": "Synthetic exceptional-writer test approval.",
        },
        "authoritative_provenance": {
            "task_key": key,
            "subject": identity,
            "authority_reference": registered["maintainer_authorization"][
                "authority_reference"
            ],
        },
    }


def subagent_advance_record(workflow, value, key="task-one", **extra):
    reference = f"writer-exception:{key}:subagent"
    if not any(item.get("reference") == reference for item in value["evidence"]):
        workflow.record_evidence(value, writer_exception_decision(value, key))
    return {
        "task_key": key,
        "writer_mode": "subagent",
        "writer_exception_reference": reference,
        **extra,
    }


def handoff(key="task-one", worker_reference=None, identity=IDENTITY, **extra):
    worker_reference = worker_reference or f"root:{key}"
    value = {
        "task_key": key,
        "worker_reference": worker_reference,
        "changed_paths": ["scripts/example.py"],
        "identity": identity,
        "route": route(),
        "acceptance_evidence": ["focused test"],
        "verification": ["unit test"],
        "applied_decisions": [
            {"task_key": key, "reference": "decision:task-one", "identity": identity}
        ],
        "verification_references": [
            {
                "task_key": key,
                "reference": "verification:unit",
                "identity": identity,
                "verification_request": verification_request(),
            }
        ],
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
    result = workflow.advance(value, {"task_key": key})
    assert result["next_action"] == "ROOT_IMPLEMENTATION_IN_PROGRESS"
    return next(
        worker for worker in value["workers"] if worker["reference"] == f"root:{key}"
    )


def dispatch_exceptional_worker(workflow, value, key="task-one"):
    to_plan(workflow, value, key)
    paths = value["tasks"][key]["owned_paths"]
    workflow.acquire_ownership(value, {"task_key": key, "paths": paths})
    workflow.advance(
        value,
        subagent_advance_record(workflow, value, key),
    )
    return workflow.register_worker(value, {"reference": "writer-one", "task_key": key})


def completed_candidate(workflow, value, key="task-one"):
    dispatch_worker(workflow, value, key)
    workflow.record_evidence(value, decision(key))
    workflow.ingest_handoff(value, handoff(key))
    workflow.record_verification(value, verification_record(key=key))
    workflow.record_verification(
        value,
        {
            "task_key": key,
            "reference": "verification:public",
            "subject": IDENTITY,
            "command": ["python", "scripts/public-safety.py"],
            "scope": ["repository"],
            "environment": {"lock": "committed"},
            "side_effects": [],
            "status": "PASS",
            "verification_type": "public_safety",
            "executor": "root",
        },
    )
    if not workflow.review_required(value["tasks"][key])["required"]:
        workflow.record_verification(
            value,
            {
                "task_key": key,
                "reference": "verification:gate",
                "subject": IDENTITY,
                **final_gate_request(),
                "status": "PASS",
                "executor": "root",
            },
        )


def test_canonical_cold_technical_review_class_is_routed():
    workflow = load_workflow_module()
    review = task(task_class="cold_technical_review")
    assert workflow.validate_task(review)["task_class"] == "cold_technical_review"


def test_authorization_phase_pairs_and_operations_fail_closed():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    with pytest.raises(workflow.WorkflowError, match="only valid in PLAN"):
        workflow.acquire_ownership(
            value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
        )
    to_plan(workflow, value)
    workflow.acquire_ownership(
        value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    with pytest.raises(
        workflow.WorkflowError, match="only valid for an implementing dispatch"
    ):
        workflow.register_worker(
            value, {"reference": "writer-one", "task_key": "task-one"}
        )
    with pytest.raises(workflow.WorkflowError, match="incompatible"):
        workflow.validate_state({**value, "phase": "VERIFY"})


def test_transition_cannot_walk_to_integration_or_grant_it_without_prerequisites():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(
        value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    workflow.advance(value, {"task_key": "task-one"})
    workflow.record_evidence(value, decision())
    workflow.ingest_handoff(value, handoff())
    with pytest.raises(workflow.WorkflowError, match="integration transition requires"):
        workflow.transition(value, "INTEGRATE")
    with pytest.raises(workflow.WorkflowError, match="verified candidate"):
        workflow.authorize(
            value,
            {"actor": "maintainer", "authorization_state": "INTEGRATION_AUTHORIZED"},
        )


def test_path_collisions_include_ancestor_descendant_and_cross_platform_normalization():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task("parent", ["scripts"]))
    workflow.register_task(value, task("child", ["scripts\\demo.py"]))
    to_plan(workflow, value, "parent")
    workflow.acquire_ownership(value, {"task_key": "parent", "paths": ["scripts"]})
    value["authorization_state"] = "MAINTAINER_AUTHORIZED"
    with pytest.raises(workflow.WorkflowError, match="path ownership collision"):
        workflow.acquire_ownership(
            value, {"task_key": "child", "paths": ["scripts/demo.py"]}
        )


def test_handoff_rejects_untyped_wrong_base_or_empty_subject_identities():
    workflow = load_workflow_module()
    for identity, message in (
        ({"base": BASE, "patch": "sha256:synthetic"}, "identity type"),
        (
            {"type": "patch", "base": "wrong-base", "patch": "sha256:synthetic"},
            "identity base",
        ),
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
    invalid["verification_references"] = [
        {"task_key": "other", "reference": "verification:unit", "identity": IDENTITY}
    ]
    with pytest.raises(workflow.WorkflowError, match="bind the handoff task"):
        workflow.ingest_handoff(value, invalid)


def test_verification_reuse_requires_typed_exact_subject_and_stable_inputs():
    workflow = load_workflow_module()
    complete = verification_record(
        untracked_inputs_relevant=True,
        untracked_inputs=[
            {
                "content_identity": "sha256:synthetic",
                "role": "fixture",
                "scope": "tests",
            }
        ],
    )
    assert workflow.verification_reuse(complete, IDENTITY, verification_request()) == {
        "reusable": True,
        "status": "REUSABLE",
    }
    assert workflow.verification_reuse(
        complete, {"type": "patch", "base": BASE, "patch": ""}, verification_request()
    ) == {"reusable": False, "status": "NON_REUSABLE"}


def test_integration_requires_matching_handoff_bound_reusable_pass_and_decision():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    completed_candidate(workflow, value)
    assert (
        workflow.advance(
            value,
            {"task_key": "task-one", "verification_request": verification_request()},
        )["next_action"]
        == "AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION"
    )
    request = {
        "actor": "maintainer",
        "authorization_state": "INTEGRATION_AUTHORIZED",
        "task_key": "task-one",
        "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0],
        "verification_reference": "verification:unit",
        "verification_request": verification_request(),
    }
    assert workflow.authorize(value, request) == {
        "authorization_state": "INTEGRATION_AUTHORIZED",
        "actor": "maintainer",
        "phase": "INTEGRATE",
    }


def test_integration_rejects_wrong_verification_identity_or_reference():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    completed_candidate(workflow, value)
    bad = {
        "actor": "maintainer",
        "authorization_state": "INTEGRATION_AUTHORIZED",
        "task_key": "task-one",
        "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0],
        "verification_reference": "other",
        "verification_request": verification_request(),
    }
    with pytest.raises(workflow.WorkflowError, match="handoff-bound verification"):
        workflow.authorize(value, bad)


def test_protected_review_repairs_continue_without_an_arbitrary_cycle_limit():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task(task_class="protected_boundary_implementation"))
    completed_candidate(workflow, value)
    record = {"task_key": "task-one", "verification_request": verification_request()}
    assert workflow.advance(value, record)["next_action"] == "AWAIT_COLD_REVIEW"
    result = workflow.advance(value, {**record, "review_status": "REPAIR_REQUIRED"})
    assert result["next_action"] == "REPAIR_IN_SCOPE"
    assert value["phase"] == "PLAN"
    assert value["authorization_state"] == "MAINTAINER_AUTHORIZED"
    review = value["reviews"][0]
    review["repair_cycles"] = 3
    review["status"] = "REQUIRED"
    value["phase"] = "VERIFY"
    value["authorization_state"] = "VERIFIED"
    result = workflow.advance(value, {**record, "review_status": "REPAIR_REQUIRED"})
    assert result["next_action"] == "REPAIR_IN_SCOPE"
    assert review["repair_cycles"] == 4


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
    value["workers"].append(
        {
            "reference": "orphan",
            "task_key": "missing",
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
    assert workflow.release_ownership(value, {"task_key": "task-one"}) == {
        "task_key": "task-one",
        "released": True,
    }


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
    value["evidence"].append(
        {"kind": "note", "subject": {"id": "synthetic"}, "transcript": "do not copy"}
    )
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
    workflow.record_verification(
        value,
        {**verification_record(key="task-two"), "reference": "verification:task-two"},
    )
    request = {
        "actor": "maintainer",
        "authorization_state": "INTEGRATION_AUTHORIZED",
        "task_key": "task-one",
        "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0],
        "verification_reference": "verification:task-two",
        "verification_request": verification_request(),
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
        "actor": "maintainer",
        "authorization_state": "INTEGRATION_AUTHORIZED",
        "task_key": "task-one",
        "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0],
        "verification_reference": "verification:unit",
        "verification_request": verification_request(),
    }
    with pytest.raises(workflow.WorkflowError, match="root-executed"):
        workflow.authorize(value, request)
    workflow.record_verification(
        value,
        {
            "task_key": "task-one",
            "reference": "verification:gate-after-review",
            "subject": IDENTITY,
            **final_gate_request(),
            "status": "PASS",
            "executor": "root",
        },
    )
    assert workflow.authorize(value, request)["phase"] == "INTEGRATE"


def test_registered_risk_trigger_cannot_be_omitted_from_advance():
    workflow = load_workflow_module()
    value = state(workflow)
    record = task()
    record["review_triggers"] = ["public_repository_safety"]
    workflow.register_task(value, record)
    completed_candidate(workflow, value)
    result = workflow.advance(
        value, {"task_key": "task-one", "verification_request": verification_request()}
    )
    assert result["next_action"] == "AWAIT_COLD_REVIEW"
    assert result["review_packet"]["subject"] == IDENTITY


def test_final_gate_is_root_executed_once_per_immutable_candidate():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    completed_candidate(workflow, value)
    with pytest.raises(workflow.WorkflowError, match="registrations are prohibited"):
        workflow.record_evidence(
            value,
            {
                "kind": "verifier_registration",
                "task_key": "task-one",
                "subject": IDENTITY,
            },
        )
    with pytest.raises(workflow.WorkflowError, match="already recorded"):
        workflow.record_verification(
            value,
            {
                "task_key": "task-one",
                "reference": "verification:duplicate-gate",
                "subject": IDENTITY,
                **final_gate_request(),
                "status": "PASS",
                "executor": "root",
            },
        )


def test_public_safety_requires_root_and_latest_pass():
    workflow = load_workflow_module()
    request = {
        "actor": "maintainer",
        "authorization_state": "INTEGRATION_AUTHORIZED",
        "task_key": "task-one",
        "identity": IDENTITY,
        "applied_decision": handoff()["applied_decisions"][0],
        "verification_reference": "verification:unit",
        "verification_request": verification_request(),
    }
    stale_safety = state(workflow)
    workflow.register_task(stale_safety, task())
    completed_candidate(workflow, stale_safety)
    workflow.record_verification(
        stale_safety,
        {
            "task_key": "task-one",
            "reference": "verification:public",
            "subject": IDENTITY,
            "command": ["python", "scripts/public-safety.py"],
            "scope": ["repository"],
            "environment": {"lock": "committed"},
            "side_effects": [],
            "status": "FAIL",
            "verification_type": "public_safety",
            "executor": "root",
        },
    )
    with pytest.raises(workflow.WorkflowError, match="public-safety"):
        workflow.authorize(stale_safety, request)


def test_authority_provenance_and_route_observation_fail_closed():
    workflow = load_workflow_module()
    invalid = task()
    del invalid["maintainer_authorization"]["readiness_evidence"]
    with pytest.raises(workflow.WorkflowError, match="provenance"):
        workflow.validate_task(invalid, exact_base=BASE)
    invalid = task()
    invalid["route"]["runtime_observation"] = {
        "availability": "OBSERVED",
        "source": "synthetic_telemetry",
        "model": "other-model",
        "effort": "high",
    }
    with pytest.raises(workflow.WorkflowError, match="compliance"):
        workflow.validate_task(invalid, exact_base=BASE)

    mismatch = task()
    mismatch["route"]["runtime_observation"] = invalid["route"]["runtime_observation"]
    mismatch["route"]["declared_route_compliance"] = "FAIL"
    value = state(workflow)
    workflow.register_task(value, mismatch)
    to_plan(workflow, value)
    workflow.acquire_ownership(
        value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    with pytest.raises(workflow.WorkflowError, match="route mismatch"):
        workflow.advance(value, {"task_key": "task-one"})


def test_route_selection_rejects_prompt_or_unknown_mechanisms():
    workflow = load_workflow_module()
    prompt_only = task()
    prompt_only["route"]["selection_mechanism"] = "prompt"
    with pytest.raises(workflow.WorkflowError, match="JSON object"):
        workflow.validate_task(prompt_only, exact_base=BASE)
    unknown = task()
    unknown["route"]["selection_mechanism"] = {
        "kind": "invented",
        "model_selector": "model",
        "effort_selector": "effort",
    }
    with pytest.raises(workflow.WorkflowError, match="host_controls"):
        workflow.validate_task(unknown, exact_base=BASE)
    missing_selector = task()
    del missing_selector["route"]["selection_mechanism"]["effort_selector"]
    with pytest.raises(workflow.WorkflowError, match="executable"):
        workflow.validate_task(missing_selector, exact_base=BASE)


def test_matched_experiment_label_requires_predeclared_pair_metadata():
    workflow = load_workflow_module()
    control = task()
    control["route"]["experiment"] = {"label": "CONTROL", "predeclared": True}
    with pytest.raises(workflow.WorkflowError, match="predeclared matched pair"):
        workflow.validate_task(control, exact_base=BASE)
    control["route"]["experiment"] = matched_experiment(
        "CONTROL",
        "pair-synthetic",
        "run-control",
        "bounded_implementation_or_repair",
    )
    assert (
        workflow.validate_task(control, exact_base=BASE)["route"]["experiment"]
        == (control["route"]["experiment"])
    )
    experimental = task()
    experimental["route"].update(
        {
            "declared_route": "sol-ultra",
            "declared_model": "gpt-5.6-sol",
            "declared_effort": "ultra",
            "selected_model": "gpt-5.6-sol",
            "selected_effort": "ultra",
            "experiment": matched_experiment(
                "EXPERIMENTAL",
                "pair-synthetic",
                "run-experimental",
                "bounded_implementation_or_repair",
            ),
        }
    )
    assert (
        workflow.validate_task(experimental, exact_base=BASE)["route"]["declared_route"]
        == "sol-ultra"
    )


def test_route_trial_registry_requires_six_isolated_scored_pairs():
    workflow = load_workflow_module()
    complete = matched_trial()
    assert workflow.validate_route_trial(complete)[
        "eligible_for_default_route_decision"
    ]
    value = state(workflow)
    result = workflow.record_route_trial(value, complete)
    assert result == {
        "recorded": "route_trial",
        "trial_id": "trial-synthetic",
        "eligible_for_default_route_decision": True,
    }
    with pytest.raises(workflow.WorkflowError, match="already registered"):
        workflow.record_route_trial(value, complete)

    missing_pair = copy.deepcopy(complete)
    missing_pair["trial_id"] = "trial-missing-pair"
    missing_pair["pairs"].pop()
    with pytest.raises(workflow.WorkflowError, match="exactly six pairs"):
        workflow.validate_route_trial(missing_pair)
    reused_context = copy.deepcopy(complete)
    reused_context["pairs"][0]["runs"][1]["context_identity"] = reused_context["pairs"][
        0
    ]["runs"][0]["context_identity"]
    with pytest.raises(workflow.WorkflowError, match="unique run/context"):
        workflow.validate_route_trial(reused_context)
    unscored = copy.deepcopy(complete)
    del unscored["pairs"][0]["runs"][0]["score"]
    with pytest.raises(workflow.WorkflowError, match="score must be a JSON object"):
        workflow.validate_route_trial(unscored)
    arbitrary_score = copy.deepcopy(complete)
    arbitrary_score["pairs"][0]["runs"][0]["score"]["components"]["quality"] = 99
    with pytest.raises(workflow.WorkflowError, match="raw scoring inputs"):
        workflow.validate_route_trial(arbitrary_score)
    different_ledgers = copy.deepcopy(complete)
    different_ledgers["pairs"][0]["runs"][1]["score"]["ledger_ref"] = "ledger:different"
    with pytest.raises(workflow.WorkflowError, match="share one ledger"):
        workflow.validate_route_trial(different_ledgers)
    mixed_zero_ledger = copy.deepcopy(complete)
    zero_runs = mixed_zero_ledger["pairs"][0]["runs"]
    for run in zero_runs:
        raw = run["score"]["raw"]
        raw["credited_ledger_weight"] = 0
        raw["total_ledger_weight"] = 0
        run["score"]["components"]["quality"] = 100
        run["score"]["components"]["total"] = 100
    zero_runs[1]["score"]["raw"]["acceptance_complete"] = False
    with pytest.raises(workflow.WorkflowError, match="both runs' acceptance"):
        workflow.validate_route_trial(mixed_zero_ledger)
    mismatched_task_class = copy.deepcopy(complete)
    mismatched_task_class["pairs"][0]["runs"][1]["task_class"] = (
        "protected_boundary_implementation"
    )
    with pytest.raises(workflow.WorkflowError, match="same frozen task class"):
        workflow.validate_route_trial(mismatched_task_class)

    observational = {
        "trial_id": "trial-observational",
        "classification": "OBSERVATIONAL",
        "eligible_for_default_route_decision": False,
        "reason": "The tasks or isolated contexts were not matched.",
    }
    assert not workflow.validate_route_trial(observational)[
        "eligible_for_default_route_decision"
    ]
    observational["eligible_for_default_route_decision"] = True
    with pytest.raises(workflow.WorkflowError, match="cannot support"):
        workflow.validate_route_trial(observational)


def test_decision_authority_must_match_registered_task_authority():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    unrelated = decision()
    unrelated["authoritative_provenance"]["authority_reference"] = "authority:unrelated"
    with pytest.raises(workflow.WorkflowError, match="authoritative"):
        workflow.record_evidence(value, unrelated)
    workflow.record_evidence(value, decision())
    value["evidence"][-1]["authoritative_provenance"]["authority_reference"] = (
        "authority:unrelated"
    )
    with pytest.raises(workflow.WorkflowError, match="authoritative"):
        workflow.validate_state(value)


def test_read_only_tasks_and_windows_absolute_paths_cannot_be_written():
    workflow = load_workflow_module()
    for path in (
        "C:\\work\\file.py",
        "C:work\\file.py",
        "\\rooted\\file.py",
        "\\\\server\\share\\file.py",
        "\\\\?\\C:\\file.py",
        "/repo/file.py",
    ):
        with pytest.raises(workflow.WorkflowError, match="unsafe owned path"):
            workflow.normalized_path(path)
    value = state(workflow)
    workflow.register_task(value, task(task_class="cold_technical_review"))
    to_plan(workflow, value)
    with pytest.raises(workflow.WorkflowError, match="read-only"):
        workflow.acquire_ownership(
            value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
        )


def test_root_writer_is_default_and_subagent_dispatch_requires_exception():
    workflow = load_workflow_module()
    routine = task(task_class="routine_administration")
    value = state(workflow)
    workflow.register_task(value, routine)
    to_plan(workflow, value)
    with pytest.raises(workflow.WorkflowError, match="read-only"):
        workflow.acquire_ownership(
            value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
        )

    root_candidate = state(workflow)
    workflow.register_task(root_candidate, task())
    to_plan(workflow, root_candidate)
    workflow.acquire_ownership(
        root_candidate, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    result = workflow.advance(root_candidate, {"task_key": "task-one"})
    assert result["root_writer"] == {"reference": "root:task-one", "attempt": 1}
    assert "context" not in result
    assert root_candidate["workers"][0]["execution_mode"] == "root"

    candidate = state(workflow)
    workflow.register_task(candidate, task())
    to_plan(workflow, candidate)
    workflow.acquire_ownership(
        candidate, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    with pytest.raises(workflow.WorkflowError, match="maintainer exception"):
        workflow.advance(candidate, {"task_key": "task-one", "writer_mode": "subagent"})
    with pytest.raises(workflow.WorkflowError, match="durable maintainer exception"):
        workflow.advance(
            candidate,
            {
                "task_key": "task-one",
                "writer_mode": "subagent",
                "maintainer_writer_exception": True,
            },
        )
    workflow.advance(
        candidate,
        subagent_advance_record(workflow, candidate),
    )
    with pytest.raises(workflow.WorkflowError, match="derived"):
        workflow.register_worker(
            candidate,
            {"reference": "writer-one", "task_key": "task-one", "writer": False},
        )
    mismatched = route()
    mismatched["selection_mechanism"]["control_surface"] = "different_host_control"
    with pytest.raises(workflow.WorkflowError, match="supplied route"):
        workflow.register_worker(
            candidate,
            {"reference": "writer-one", "task_key": "task-one", "route": mismatched},
        )


def test_second_root_writer_requires_a_durable_maintainer_exception():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task("task-one", ["scripts/one.py"]))
    workflow.register_task(value, task("task-two", ["scripts/two.py"]))
    to_plan(workflow, value)
    for key in ("task-one", "task-two"):
        workflow.acquire_ownership(
            value, {"task_key": key, "paths": value["tasks"][key]["owned_paths"]}
        )
    workflow.advance(value, {"task_key": "task-one"})
    guards = {name: True for name in workflow.REQUIRED_SECOND_WRITER_GUARDS}
    record = {"task_key": "task-two", "independence_guards": guards}
    with pytest.raises(workflow.WorkflowError, match="durable maintainer exception"):
        workflow._activate_root_writer(value, value["tasks"]["task-two"], record)
    exception = writer_exception_decision(value, "task-two", "root")
    workflow.record_evidence(value, exception)
    record["writer_exception_reference"] = exception["reference"]
    second = workflow._activate_root_writer(value, value["tasks"]["task-two"], record)
    assert second["writer_exception_reference"] == exception["reference"]
    assert workflow.validate_state(value) is value
    del second["writer_exception_reference"]
    with pytest.raises(workflow.WorkflowError, match="durable maintainer exception"):
        workflow.validate_state(value)


def test_guarded_second_writer_is_reachable_and_third_or_unguarded_writer_is_rejected():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task("task-one", ["scripts/one.py"]))
    workflow.register_task(value, task("task-two", ["scripts/two.py"]))
    workflow.register_task(value, task("task-three", ["scripts/three.py"]))
    to_plan(workflow, value)
    for key in ("task-one", "task-two", "task-three"):
        workflow.acquire_ownership(
            value, {"task_key": key, "paths": value["tasks"][key]["owned_paths"]}
        )
    workflow.advance(value, {"task_key": "task-one"})
    with pytest.raises(workflow.WorkflowError, match="durable task-specific"):
        workflow.register_worker(
            value, {"reference": "writer-two", "task_key": "task-two"}
        )
    with pytest.raises(workflow.WorkflowError, match="independence guard"):
        workflow.advance(
            value,
            subagent_advance_record(workflow, value, "task-two"),
        )
    guards = {name: True for name in workflow.REQUIRED_SECOND_WRITER_GUARDS}
    assert (
        workflow.advance(
            value,
            subagent_advance_record(
                workflow, value, "task-two", independence_guards=guards
            ),
        )["advanced"]
        is True
    )
    assert (
        workflow.register_worker(
            value, {"reference": "writer-two", "task_key": "task-two"}
        )["status"]
        == "ACTIVE"
    )
    del value["workers"][1]["independence_guards"]
    with pytest.raises(workflow.WorkflowError, match="independence guard"):
        workflow.validate_state(value)
    value["workers"][1]["independence_guards"] = None
    with pytest.raises(workflow.WorkflowError, match="independence guard"):
        workflow.validate_state(value)
    value["workers"][1]["independence_guards"] = guards
    with pytest.raises(workflow.WorkflowError, match="third"):
        workflow.advance(
            value,
            subagent_advance_record(
                workflow, value, "task-three", independence_guards=guards
            ),
        )


def test_duplicate_task_and_ambiguous_spawn_are_rejected_or_durable():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(
        value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    workflow.advance(
        value,
        subagent_advance_record(workflow, value),
    )
    value["workers"].append(
        {
            "reference": "duplicate",
            "task_key": "task-one",
            "task_class": "normal_implementation",
            "writer": True,
            "status": "ACTIVE",
            "route": task()["route"],
            "attempt": 2,
            "monitor": workflow._new_monitor("ACTIVE"),
        }
    )
    with pytest.raises(workflow.WorkflowError, match="for a task"):
        workflow.validate_state(value)
    value["workers"].pop()
    assert (
        workflow.reconcile_worker(
            value,
            {
                "reference": "manual:task-one",
                "status": "SPAWN_UNKNOWN",
                "ambiguity_reference": "ambiguity:synthetic",
            },
        )["status"]
        == "SPAWN_UNKNOWN"
    )
    assert (
        workflow.reconcile_worker(value, {"reference": "manual:task-one"})["retry"]
        == "BLOCKED"
    )
    value["workers"][0]["ambiguity_reference"] = ""
    with pytest.raises(workflow.WorkflowError, match="ambiguity_reference"):
        workflow.validate_state(value)


def test_persisted_verifier_and_duplicate_gate_records_fail_validation():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    completed_candidate(workflow, value)
    value["evidence"].append(
        {
            "kind": "verifier_registration",
            "task_key": "task-one",
            "subject": IDENTITY,
        }
    )
    with pytest.raises(workflow.WorkflowError, match="registrations are prohibited"):
        workflow.validate_state(value)

    duplicate = state(workflow)
    workflow.register_task(duplicate, task())
    completed_candidate(workflow, duplicate)
    second_gate = dict(duplicate["verification"][-1])
    second_gate["reference"] = "verification:duplicate-gate"
    second_gate["sequence"] += 1
    duplicate["verification"].append(second_gate)
    with pytest.raises(workflow.WorkflowError, match="duplicate full gate"):
        workflow.validate_state(duplicate)


def test_newer_fail_invalidates_only_its_task_verification_pass():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    workflow.register_task(value, task("task-two", ["tests/example.py"]))
    completed_candidate(workflow, value)
    workflow.record_verification(value, {**verification_record(), "status": "FAIL"})
    assert value["authorization_state"] == "IMPLEMENTING"
    assert (
        workflow.advance(
            value,
            {"task_key": "task-one", "verification_request": verification_request()},
        )["next_action"]
        == "RUN_OR_RECORD_REQUIRED_VERIFICATION"
    )
    different_request = state(workflow)
    workflow.register_task(different_request, task())
    completed_candidate(workflow, different_request)
    workflow.record_verification(
        different_request,
        {
            "task_key": "task-one",
            "reference": "verification:other",
            "subject": IDENTITY,
            "command": ["python", "-m", "pytest", "tests/other"],
            "scope": ["tests/other"],
            "environment": {"lock": "committed"},
            "side_effects": [],
            "status": "FAIL",
        },
    )
    assert different_request["authorization_state"] == "VERIFIED"
    assert (
        workflow._reusable_verification(
            different_request, "task-one", IDENTITY, verification_request()
        )["status"]
        == "PASS"
    )
    separate = state(workflow)
    workflow.register_task(separate, task())
    workflow.register_task(separate, task("task-two", ["tests/example.py"]))
    completed_candidate(workflow, separate)
    workflow.record_verification(
        separate,
        {
            **verification_record(key="task-two"),
            "reference": "verification:task-two",
            "status": "FAIL",
        },
    )
    assert separate["authorization_state"] == "VERIFIED"
    assert (
        workflow._reusable_verification(
            separate, "task-one", IDENTITY, verification_request()
        )["status"]
        == "PASS"
    )


def test_worker_heartbeat_backs_off_without_replaying_context_or_spawning_duplicates():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(
        value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    workflow.advance(
        value,
        subagent_advance_record(workflow, value),
    )
    first = workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 100})
    assert first == {
        "action": "WAIT_UNTIL_HEARTBEAT",
        "reference": "manual:task-one",
        "interval_seconds": 120,
        "next_check_at": 220,
    }
    early = workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 150})
    assert early == first
    backed_off = workflow.monitor_worker(
        value, {"reference": "manual:task-one", "now": 220}
    )
    assert backed_off["action"] == "WAIT_UNTIL_HEARTBEAT"
    assert backed_off["interval_seconds"] == 240
    assert len(value["workers"]) == 1
    assert (
        workflow.advance(value, {"task_key": "task-one"})["next_action"]
        == "AWAIT_MANUAL_WORKER_REFERENCE"
    )
    assert len(value["workers"]) == 1


def test_worker_heartbeat_resets_on_change_and_wakes_immediately_for_terminal_events():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    dispatch_exceptional_worker(workflow, value)
    workflow.monitor_worker(value, {"reference": "writer-one", "now": 0})
    workflow.monitor_worker(value, {"reference": "writer-one", "now": 120})
    changed = workflow.monitor_worker(
        value,
        {"reference": "writer-one", "now": 121, "output_identity": "output:synthetic"},
    )
    assert changed["action"] == "HEARTBEAT_CHANGED"
    assert changed["interval_seconds"] == 120
    assert (
        workflow.monitor_worker(
            value, {"reference": "writer-one", "now": 122, "system_error": True}
        )["reason"]
        == "SYSTEM_ERROR"
    )
    assert (
        workflow.monitor_worker(
            value, {"reference": "writer-one", "now": 123, "maintainer_input": True}
        )["reason"]
        == "MAINTAINER_INPUT"
    )
    completed = workflow.monitor_worker(
        value, {"reference": "writer-one", "now": 124, "status": "COMPLETE"}
    )
    assert completed == {
        "action": "WAKE_IMMEDIATELY",
        "reason": "COMPLETION",
        "reference": "writer-one",
    }


def test_worker_heartbeat_requires_durable_ambiguous_spawn_reconciliation():
    workflow = load_workflow_module()
    value = state(workflow)
    workflow.register_task(value, task())
    to_plan(workflow, value)
    workflow.acquire_ownership(
        value, {"task_key": "task-one", "paths": ["scripts/example.py"]}
    )
    workflow.advance(
        value,
        subagent_advance_record(workflow, value),
    )
    with pytest.raises(workflow.WorkflowError, match="ambiguity_reference"):
        workflow.monitor_worker(
            value, {"reference": "manual:task-one", "now": 0, "status": "SPAWN_UNKNOWN"}
        )
    immediate = workflow.monitor_worker(
        value,
        {
            "reference": "manual:task-one",
            "now": 0,
            "status": "SPAWN_UNKNOWN",
            "ambiguity_reference": "ambiguity:synthetic",
        },
    )
    assert immediate["reason"] == "AMBIGUOUS_SPAWN_RECONCILIATION"
    assert (
        workflow.monitor_worker(value, {"reference": "manual:task-one", "now": 1})[
            "reason"
        ]
        == "AMBIGUOUS_SPAWN_RECONCILIATION"
    )


def test_ci_runs_prs_once_and_keeps_main_and_two_os_coverage():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    workflow_text = (root / ".github" / "workflows" / "temper-gate.yml").read_text(
        encoding="utf-8"
    )
    assert "push:\n    branches: [main]" in workflow_text
    assert "pull_request:" in workflow_text
    assert (
        "group: temper-gate-${{ github.workflow }}-${{ github.ref }}" in workflow_text
    )
    assert "cancel-in-progress: true" in workflow_text
    assert "os: [ubuntu-latest, windows-latest]" in workflow_text


def test_route_trial_policy_is_explicit_and_matched():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    policy = (
        root / "docs" / "workflow" / "policies" / "model-routing-policy.yaml"
    ).read_text(encoding="utf-8")
    procedure = (
        root / "docs" / "workflow" / "procedures" / "model-route-and-experiment.md"
    ).read_text(encoding="utf-8")
    assert "matched_pairs: 6" in policy
    assert "runs_per_pair: 2" in policy
    assert "total_runs: 12" in policy
    assert "isolated worktrees" in procedure
    assert "label the comparison `OBSERVATIONAL`" in procedure
    assert "total = 0.50 * quality" in procedure
