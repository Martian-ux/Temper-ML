#!/usr/bin/env python3
"""Fail-closed, JSON-only coordinator for bounded Temper ML workflow tasks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import PurePosixPath
from typing import Any


class WorkflowError(ValueError):
    """A record is incomplete, unsafe, or requests an illegal transition."""


PHASES = (
    "RECONCILE",
    "DELIBERATE",
    "DECIDE",
    "PLAN",
    "DISPATCH",
    "VERIFY",
    "INTEGRATE",
    "CLOSE",
)
ALLOWED_TRANSITIONS = {
    "RECONCILE": {"DELIBERATE"},
    "DELIBERATE": {"DECIDE"},
    "DECIDE": {"PLAN"},
    "PLAN": {"DISPATCH"},
    "DISPATCH": {"VERIFY", "RECONCILE"},
    "VERIFY": {"PLAN", "INTEGRATE", "RECONCILE"},
    "INTEGRATE": {"PLAN", "VERIFY", "DECIDE", "RECONCILE", "CLOSE"},
    "CLOSE": set(),
}
ACTIVE_WORKER_STATES = {"SPAWN_REQUESTED", "SPAWN_UNKNOWN", "ACTIVE", "WAITING", "BLOCKED"}
REQUIRED_SECOND_WRITER_GUARDS = {
    "disjoint_paths",
    "independent_acceptance_criteria",
    "no_uncommitted_output_dependency",
    "no_shared_blocking_decision",
    "known_integration_order",
    "explicit_recorded_approval",
}
PUBLIC_ROUTE = {
    "routine_administration": ("luna_or_cheapest", {"low", "medium"}),
    "mechanical_change": ("terra-medium", {"medium"}),
    "normal_implementation": ("terra-high", {"high"}),
    "protected_boundary_implementation": ("terra-high", {"high"}),
    "technical_review": ("terra-high", {"high"}),
}
REQUIRED_TASK_FIELDS = {
    "task_key",
    "task_class",
    "objective",
    "exact_base",
    "owned_paths",
    "acceptance_criteria",
    "non_goals",
    "verification",
    "review",
    "route",
    "stop_conditions",
    "authorization_state",
    "classification",
    "mission_ref",
    "mission_fit",
}


def default_state() -> dict[str, Any]:
    """Return a JSON-compatible empty coordinator record."""

    return {
        "schema_version": 1,
        "phase": "RECONCILE",
        "authorization_state": "PROPOSED",
        "mission": {"priority": "", "mission_fit": "PENDING"},
        "repository": {"exact_base": ""},
        "tasks": {},
        "workers": [],
        "ownership": [],
        "evidence": [],
        "verification": [],
        "checkpoints": [],
        "handoffs": [],
        "reviews": [],
    }


def _as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must be a JSON object")
    return value


def _as_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise WorkflowError(f"{label} must be a JSON array")
    return value


def normalized_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowError("owned path must be a non-empty repository-relative string")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path == ".":
        raise WorkflowError(f"unsafe owned path: {value!r}")
    return path.as_posix()


def validate_route(route: Any, task_class: str | None = None) -> None:
    route = _as_object(route, "route")
    selected_model = route.get("selected_model")
    selected_effort = route.get("selected_effort")
    if not isinstance(selected_model, str) or not selected_model:
        raise WorkflowError("selected model is required; prompt text is not route evidence")
    if not isinstance(selected_effort, str) or not selected_effort:
        raise WorkflowError("selected reasoning effort is required; prompt text is not route evidence")
    for observed in ("observed_model", "observed_effort"):
        value = route.get(observed, "UNVERIFIED")
        if not isinstance(value, str) or not value:
            raise WorkflowError(f"{observed} must be a value or UNVERIFIED")
    if task_class in PUBLIC_ROUTE:
        expected_model, efforts = PUBLIC_ROUTE[task_class]
        if selected_model != expected_model or selected_effort not in efforts:
            raise WorkflowError(f"route is not permitted for task class {task_class}")


def validate_task(task: Any, *, exact_base: str | None = None) -> dict[str, Any]:
    task = _as_object(task, "task")
    missing = sorted(REQUIRED_TASK_FIELDS - set(task))
    if missing:
        raise WorkflowError(f"task is missing required fields: {', '.join(missing)}")
    key = task["task_key"]
    if not isinstance(key, str) or not key.strip():
        raise WorkflowError("task_key must be a non-empty string")
    if task["task_class"] not in PUBLIC_ROUTE:
        raise WorkflowError("unsupported task class")
    if task["mission_fit"] != "FIT":
        raise WorkflowError("task does not fit the active mission")
    if task["authorization_state"] != "MAINTAINER_AUTHORIZED":
        raise WorkflowError("task is not maintainer-authorized")
    if exact_base is not None and task["exact_base"] != exact_base:
        raise WorkflowError("task exact base does not match authoritative state")
    owned_paths = [normalized_path(path) for path in _as_list(task["owned_paths"], "owned_paths")]
    if not owned_paths or len(owned_paths) != len(set(owned_paths)):
        raise WorkflowError("task must own a non-empty, duplicate-free exact path list")
    if task.get("recursive_delegation") not in (None, False):
        raise WorkflowError("recursive delegation cannot be enabled")
    for name in ("acceptance_criteria", "non_goals", "verification", "review", "stop_conditions"):
        if not _as_list(task[name], name):
            raise WorkflowError(f"task {name} must not be empty")
    validate_route(task["route"], task["task_class"])
    result = copy.deepcopy(task)
    result["owned_paths"] = owned_paths
    return result


def validate_state(state: Any) -> dict[str, Any]:
    state = _as_object(state, "state")
    if state.get("schema_version") != 1:
        raise WorkflowError("unsupported or missing schema_version")
    if state.get("phase") not in PHASES:
        raise WorkflowError("unknown workflow phase")
    if state.get("authorization_state") not in {
        "PROPOSED",
        "REVIEWED_WITH_CORRECTIONS",
        "MAINTAINER_AUTHORIZED",
        "IMPLEMENTATION_READY",
        "IMPLEMENTING",
        "VERIFIED",
        "INTEGRATION_AUTHORIZED",
        "INTEGRATED",
    }:
        raise WorkflowError("unknown authorization state")
    mission = _as_object(state.get("mission"), "mission")
    if not isinstance(mission.get("priority"), str) or not mission["priority"]:
        raise WorkflowError("active mission priority is required")
    if mission.get("mission_fit") != "FIT":
        raise WorkflowError("active mission fit must be FIT")
    repository = _as_object(state.get("repository"), "repository")
    if not isinstance(repository.get("exact_base"), str) or not repository["exact_base"]:
        raise WorkflowError("authoritative exact base is required")
    tasks = _as_object(state.get("tasks"), "tasks")
    for key, task in tasks.items():
        if key != validate_task(task, exact_base=repository["exact_base"])["task_key"]:
            raise WorkflowError("task registry key does not match task_key")
    ownership = _as_list(state.get("ownership"), "ownership")
    active_paths: set[str] = set()
    active_keys: set[str] = set()
    for lease in ownership:
        lease = _as_object(lease, "ownership lease")
        if lease.get("active") is not True:
            continue
        key = lease.get("task_key")
        if key not in tasks:
            raise WorkflowError("active ownership lease has no registered task")
        if key in active_keys:
            raise WorkflowError("duplicate active ownership task key")
        active_keys.add(key)
        paths = [normalized_path(path) for path in _as_list(lease.get("paths"), "ownership paths")]
        overlap = active_paths.intersection(paths)
        if overlap:
            raise WorkflowError(f"active ownership path collision: {sorted(overlap)!r}")
        active_paths.update(paths)
    workers = _as_list(state.get("workers"), "workers")
    seen_worker_refs: set[str] = set()
    for worker in workers:
        worker = _as_object(worker, "worker")
        reference = worker.get("reference")
        if not isinstance(reference, str) or not reference:
            raise WorkflowError("worker reference is required")
        if reference in seen_worker_refs:
            raise WorkflowError("duplicate worker reference")
        seen_worker_refs.add(reference)
        if worker.get("status") not in ACTIVE_WORKER_STATES | {"COMPLETE", "RETIRED"}:
            raise WorkflowError("unknown worker status")
        if worker.get("writer") and worker.get("status") in ACTIVE_WORKER_STATES:
            key = worker.get("task_key")
            task = tasks.get(key)
            if task is None:
                raise WorkflowError("active writer has no registered matching task")
            if worker.get("task_class") != task["task_class"]:
                raise WorkflowError("active writer task class does not match its task")
            validate_route(worker.get("route"), worker["task_class"])
            if any(worker["route"].get(field) != task["route"].get(field) for field in ("selected_model", "selected_effort")):
                raise WorkflowError("active writer selected route does not match its task")
            if key not in active_keys:
                raise WorkflowError("active writer has no active ownership lease")
    if _active_writer_count(state) > 2:
        raise WorkflowError("a third active implementation writer is prohibited")
    for field in ("evidence", "verification", "checkpoints", "handoffs", "reviews"):
        _as_list(state.get(field), field)
    review_keys: set[str] = set()
    for review in state["reviews"]:
        review = _as_object(review, "review")
        key = review.get("task_key")
        if key not in tasks or key in review_keys:
            raise WorkflowError("review must bind one registered task")
        review_keys.add(key)
        if review.get("status") not in {"REQUIRED", "REPAIR_IN_PROGRESS", "REPAIR_LIMIT_REACHED", "PASS"}:
            raise WorkflowError("unknown review status")
        if review.get("reviewers") != 1 or review.get("repair_cycles") not in {0, 1}:
            raise WorkflowError("review must retain one reviewer and at most one repair cycle")
        if not isinstance(review.get("subject"), dict):
            raise WorkflowError("review must bind an exact subject")
        if review.get("route") != "terra-high" or review.get("write_ownership") != "none":
            raise WorkflowError("review must use the cold Terra-high read-only route")
        if not isinstance(review.get("rubric_ref"), str) or not review["rubric_ref"]:
            raise WorkflowError("review rubric reference is required")
    return state


def transition(state: dict[str, Any], target: str) -> dict[str, Any]:
    validate_state(state)
    if target not in ALLOWED_TRANSITIONS[state["phase"]]:
        raise WorkflowError(f"invalid transition {state['phase']} -> {target}")
    state["phase"] = target
    return {"phase": target}


def authorize(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if record.get("actor") != "maintainer":
        raise WorkflowError("only a maintainer may authorize")
    target = record.get("authorization_state", "MAINTAINER_AUTHORIZED")
    if target not in {"MAINTAINER_AUTHORIZED", "INTEGRATION_AUTHORIZED"}:
        raise WorkflowError("only maintainer authorization states may be granted")
    if target == "INTEGRATION_AUTHORIZED" and state["phase"] != "INTEGRATE":
        raise WorkflowError("integration authorization is only valid in INTEGRATE")
    state["authorization_state"] = target
    return {"authorization_state": target, "actor": "maintainer"}


def register_task(state: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    task = validate_task(task, exact_base=state["repository"]["exact_base"])
    if state["authorization_state"] != "MAINTAINER_AUTHORIZED":
        raise WorkflowError("authoritative state is not maintainer-authorized")
    key = task["task_key"]
    if key in state["tasks"]:
        raise WorkflowError("task key is already registered")
    state["tasks"][key] = task
    return {"task_key": key, "validated": True}


def _require_implementation_authorization(state: dict[str, Any]) -> None:
    if state["authorization_state"] != "MAINTAINER_AUTHORIZED":
        raise WorkflowError("authoritative state is not currently maintainer-authorized")


def acquire_ownership(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    _require_implementation_authorization(state)
    key = record.get("task_key")
    task = state["tasks"].get(key)
    if task is None:
        raise WorkflowError("ownership requires a registered task")
    paths = [normalized_path(path) for path in _as_list(record.get("paths"), "paths")]
    if set(paths) != set(task["owned_paths"]):
        raise WorkflowError("ownership paths must exactly match the task paths")
    for lease in state["ownership"]:
        if lease.get("active") and (lease.get("task_key") == key or set(lease["paths"]).intersection(paths)):
            raise WorkflowError("task-key or exact-path ownership collision")
    lease = {"task_key": key, "paths": paths, "active": True}
    state["ownership"].append(lease)
    return lease


def release_ownership(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    key = record.get("task_key")
    if any(worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES for worker in state["workers"]):
        raise WorkflowError("ownership cannot be released while a task worker is active")
    for lease in state["ownership"]:
        if lease.get("task_key") == key and lease.get("active"):
            lease["active"] = False
            return {"task_key": key, "released": True}
    raise WorkflowError("no active ownership lease for task")


def _active_writer_count(state: dict[str, Any]) -> int:
    return sum(worker.get("writer") and worker.get("status") in ACTIVE_WORKER_STATES for worker in state["workers"])


def _check_writer_capacity(state: dict[str, Any], record: dict[str, Any]) -> None:
    if not record.get("writer", True):
        return
    count = _active_writer_count(state)
    if count >= 2:
        raise WorkflowError("a third active implementation writer is prohibited")
    if count == 1:
        guards = record.get("independence_guards")
        if not isinstance(guards, dict) or not all(guards.get(name) is True for name in REQUIRED_SECOND_WRITER_GUARDS):
            raise WorkflowError("second writer requires every recorded independence guard")


def register_worker(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    _require_implementation_authorization(state)
    key = record.get("task_key")
    if key not in state["tasks"]:
        raise WorkflowError("worker requires a registered task")
    if not any(lease.get("active") and lease.get("task_key") == key for lease in state["ownership"]):
        raise WorkflowError("worker requires an active ownership lease")
    reference = record.get("reference")
    if not isinstance(reference, str) or not reference:
        raise WorkflowError("worker reference is required")
    if any(worker["reference"] == reference for worker in state["workers"]):
        raise WorkflowError("worker reference is already registered")
    if any(worker.get("task_key") == key and worker.get("status") == "SPAWN_UNKNOWN" for worker in state["workers"]):
        raise WorkflowError("SPAWN_UNKNOWN blocks retry until a maintainer replacement decision")
    manual_intent = next(
        (
            worker
            for worker in state["workers"]
            if worker.get("task_key") == key
            and worker.get("status") == "SPAWN_REQUESTED"
            and worker.get("reference") == f"manual:{key}"
        ),
        None,
    )
    if manual_intent is not None:
        if any(worker is not manual_intent and worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES for worker in state["workers"]):
            raise WorkflowError("task already has an active worker")
        manual_intent["reference"] = reference
        manual_intent["status"] = "ACTIVE"
        return manual_intent
    if any(worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES for worker in state["workers"]):
        raise WorkflowError("task already has an active worker")
    _check_writer_capacity(state, record)
    task = state["tasks"][key]
    worker = {
        "reference": reference,
        "task_key": key,
        "task_class": task["task_class"],
        "writer": record.get("writer", True),
        "status": "ACTIVE",
        "route": copy.deepcopy(task["route"]),
    }
    state["workers"].append(worker)
    return worker


def reconcile_worker(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    reference = record.get("reference")
    for worker in state["workers"]:
        if worker.get("reference") != reference:
            continue
        if worker.get("status") == "SPAWN_UNKNOWN":
            if record.get("maintainer_replacement_decision") is not True:
                return {"reference": reference, "status": "SPAWN_UNKNOWN", "retry": "BLOCKED"}
            worker["status"] = "RETIRED"
            return {"reference": reference, "status": "RETIRED", "replacement": "MAINTAINER_APPROVED"}
        return {"reference": reference, "status": worker["status"], "retry": "NOT_NEEDED"}
    raise WorkflowError("unknown worker reference")


def record_evidence(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if not isinstance(record.get("kind"), str) or not record["kind"]:
        raise WorkflowError("evidence kind is required")
    if not isinstance(record.get("subject"), dict):
        raise WorkflowError("evidence subject is required")
    state["evidence"].append(copy.deepcopy(record))
    return {"recorded": "evidence", "count": len(state["evidence"])}


def record_verification(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    for name in ("subject", "command", "scope", "environment", "side_effects", "status"):
        if name not in record:
            raise WorkflowError(f"verification {name} is required")
    if record["status"] not in {"PASS", "FAIL", "UNKNOWN", "NON_REUSABLE"}:
        raise WorkflowError("unknown verification status")
    state["verification"].append(copy.deepcopy(record))
    return {"recorded": "verification", "count": len(state["verification"])}


def verification_reuse(record: dict[str, Any], subject: dict[str, Any], requested: Any = None) -> dict[str, Any]:
    record = _as_object(record, "verification record")
    subject = _as_object(subject, "subject")
    required = ("subject", "command", "scope", "environment", "side_effects", "status")
    if any(name not in record for name in required) or record.get("status") != "PASS":
        return {"reusable": False, "status": "NON_REUSABLE"}
    if record["subject"] != subject or record.get("invalidated"):
        return {"reusable": False, "status": "NON_REUSABLE"}
    if not isinstance(requested, dict) or any(
        name not in requested or record[name] != requested[name]
        for name in ("command", "scope", "environment", "side_effects")
    ):
        return {"reusable": False, "status": "NON_REUSABLE"}
    inputs = record.get("untracked_inputs", [])
    if not isinstance(inputs, list):
        return {"reusable": False, "status": "NON_REUSABLE"}
    if record.get("untracked_inputs_relevant") and not inputs:
        return {"reusable": False, "status": "NON_REUSABLE"}
    for item in inputs:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key] for key in ("content_identity", "role", "scope")):
            return {"reusable": False, "status": "NON_REUSABLE"}
    return {"reusable": True, "status": "REUSABLE"}


def checkpoint(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if not isinstance(record.get("next_action"), str) or not record["next_action"]:
        raise WorkflowError("checkpoint requires an exact next_action")
    entry = {"phase": state["phase"], "next_action": record["next_action"]}
    state["checkpoints"].append(entry)
    return entry


def compile_context(state: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    return {
        "mission": {"priority": state["mission"]["priority"], "mission_fit": state["mission"]["mission_fit"]},
        "phase": state["phase"],
        "repository": {"exact_base": state["repository"]["exact_base"]},
        "tasks": [{"task_key": task["task_key"], "objective": task["objective"], "owned_paths": task["owned_paths"], "stop_conditions": task["stop_conditions"]} for task in state["tasks"].values()],
        "active_ownership": [lease for lease in state["ownership"] if lease.get("active")],
        "active_workers": [{"reference": worker["reference"], "task_key": worker["task_key"], "status": worker["status"]} for worker in state["workers"] if worker["status"] in ACTIVE_WORKER_STATES],
        "verification": [{"status": item.get("status"), "subject": item.get("subject")} for item in state["verification"]],
        "next_action": next_action(state),
    }


def ingest_handoff(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    key = record.get("task_key")
    task = state["tasks"].get(key)
    if task is None:
        raise WorkflowError("handoff task is not registered")
    worker_reference = record.get("worker_reference")
    if not isinstance(worker_reference, str) or not worker_reference:
        raise WorkflowError("handoff worker_reference is required")
    worker = next((worker for worker in state["workers"] if worker.get("reference") == worker_reference and worker.get("task_key") == key), None)
    if worker is None:
        raise WorkflowError("handoff worker reference does not belong to the task")
    if worker.get("status") != "ACTIVE":
        raise WorkflowError("handoff worker is not active")
    changed = [normalized_path(path) for path in _as_list(record.get("changed_paths"), "changed_paths")]
    if not set(changed).issubset(task["owned_paths"]):
        raise WorkflowError("handoff changed paths exceed task ownership")
    for name in ("identity", "route", "acceptance_evidence", "verification", "scope_safety", "open_findings", "integration_guidance"):
        if name not in record:
            raise WorkflowError(f"handoff {name} is required")
    if not _as_list(record["acceptance_evidence"], "handoff acceptance_evidence"):
        raise WorkflowError("handoff acceptance_evidence must not be empty")
    if not _as_list(record["verification"], "handoff verification"):
        raise WorkflowError("handoff verification must not be empty")
    validate_route(record["route"], task["task_class"])
    if any(record["route"].get(field) != task["route"].get(field) for field in ("selected_model", "selected_effort")):
        raise WorkflowError("handoff selected route does not match its task")
    worker["status"] = "COMPLETE"
    state["handoffs"].append(copy.deepcopy(record))
    if state["phase"] == "DISPATCH":
        transition(state, "VERIFY")
    return {"task_key": key, "worker_reference": worker_reference, "ingested": True}


def review_required(task: dict[str, Any], triggers: list[str]) -> dict[str, Any]:
    task = validate_task(task)
    if task["task_class"] == "protected_boundary_implementation":
        return {"required": True, "reviewers": 1, "route": "terra-high", "reason": "protected_boundary"}
    return {
        "required": bool(triggers),
        "reviewers": 1 if triggers else 0,
        "route": "terra-high" if triggers else None,
        "reason": "risk_trigger" if triggers else "no_matrix_trigger",
    }


def next_action(state: dict[str, Any]) -> str:
    if state["authorization_state"] != "MAINTAINER_AUTHORIZED" and state["phase"] != "INTEGRATE":
        return "STOP_FOR_MAINTAINER_AUTHORIZATION"
    if any(worker.get("status") == "SPAWN_UNKNOWN" for worker in state["workers"]):
        return "RECONCILE_AMBIGUOUS_SPAWN"
    actions = {
        "RECONCILE": "VALIDATE_AUTHORITATIVE_STATE",
        "DELIBERATE": "DETERMINE_NEXT_LEGAL_TRANSITION",
        "DECIDE": "RECORD_DECISION_OR_STOP",
        "PLAN": "PREPARE_MANUAL_LAUNCH_PACKET",
        "DISPATCH": "AWAIT_MANUAL_WORKER_REFERENCE",
        "VERIFY": "REUSE_OR_RUN_REQUIRED_VERIFICATION",
        "INTEGRATE": "AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION",
        "CLOSE": "STOP",
    }
    return actions[state["phase"]]


def _advance_result(state: dict[str, Any], *, advanced: bool, next_step: str, transitions: list[str], **extra: Any) -> dict[str, Any]:
    result = {
        "advanced": advanced,
        "next_action": next_step,
        "automatic_transitions": transitions,
        "context": compile_context(state),
    }
    result.update(extra)
    return result


def _latest_handoff(state: dict[str, Any], task_key: str) -> dict[str, Any] | None:
    return next((handoff for handoff in reversed(state["handoffs"]) if handoff.get("task_key") == task_key), None)


def _review_for_task(state: dict[str, Any], task_key: str) -> dict[str, Any] | None:
    return next((review for review in state["reviews"] if review.get("task_key") == task_key), None)


def _reusable_verification(
    state: dict[str, Any], subject: dict[str, Any], requested: Any
) -> dict[str, Any] | None:
    for verification in reversed(state["verification"]):
        if verification_reuse(verification, subject, requested)["reusable"]:
            return verification
    return None


def _integration_evidence(
    task_key: str,
    handoff: dict[str, Any],
    verification: dict[str, Any],
    review: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_key": task_key,
        "identity": handoff["identity"],
        "verification": copy.deepcopy(verification),
        "scope_safety": handoff["scope_safety"],
        "review": review["status"] if review is not None else "NOT_REQUIRED",
    }


def advance(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if state["phase"] == "INTEGRATE":
        return _advance_result(state, advanced=False, next_step="AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION", transitions=[])
    if state["phase"] == "CLOSE":
        return _advance_result(state, advanced=False, next_step="STOP", transitions=[])
    if state["authorization_state"] != "MAINTAINER_AUTHORIZED":
        return _advance_result(state, advanced=False, next_step="STOP_FOR_MAINTAINER_AUTHORIZATION", transitions=[])
    if any(worker.get("status") == "SPAWN_UNKNOWN" for worker in state["workers"]):
        return _advance_result(state, advanced=False, next_step="RECONCILE_AMBIGUOUS_SPAWN", transitions=[])
    if state["phase"] == "DISPATCH":
        return _advance_result(state, advanced=False, next_step="AWAIT_MANUAL_WORKER_REFERENCE", transitions=[])

    key = record.get("task_key")
    task = state["tasks"].get(key)
    if task is None:
        raise WorkflowError("advance requires a registered task_key")
    transitions: list[str] = []
    while state["phase"] in {"RECONCILE", "DELIBERATE", "DECIDE"}:
        target = next(iter(ALLOWED_TRANSITIONS[state["phase"]]))
        transition(state, target)
        transitions.append(target)

    if state["phase"] == "PLAN":
        _require_implementation_authorization(state)
        if not any(lease.get("active") and lease.get("task_key") == key for lease in state["ownership"]):
            raise WorkflowError("dispatch requires an active ownership lease")
        if any(worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES for worker in state["workers"]):
            raise WorkflowError("task already has an active worker")
        validate_route(task["route"], task["task_class"])
        _check_writer_capacity(state, {"writer": True})
        intent = {
            "reference": f"manual:{key}",
            "task_key": key,
            "task_class": task["task_class"],
            "writer": True,
            "status": "SPAWN_REQUESTED",
            "route": copy.deepcopy(task["route"]),
        }
        if any(worker["reference"] == intent["reference"] for worker in state["workers"]):
            raise WorkflowError("manual dispatch intent already exists")
        state["workers"].append(intent)
        transition(state, "DISPATCH")
        transitions.append("DISPATCH")
        return _advance_result(
            state,
            advanced=True,
            next_step="AWAIT_MANUAL_WORKER_REFERENCE",
            transitions=transitions,
            manual_launch_packet={"task_key": key, "owned_paths": task["owned_paths"], "selected_route": task["route"], "manual_adapter": True},
        )

    if state["phase"] != "VERIFY":
        raise WorkflowError("advance reached an unsupported phase")
    handoff = _latest_handoff(state, key)
    if handoff is None:
        return _advance_result(state, advanced=False, next_step="AWAIT_HANDOFF", transitions=transitions)
    subject = _as_object(handoff.get("identity"), "handoff identity")
    requested_verification = record.get("verification_request")
    reusable_verification = _reusable_verification(state, subject, requested_verification)
    if reusable_verification is None:
        return _advance_result(
            state,
            advanced=False,
            next_step="RUN_OR_RECORD_REQUIRED_VERIFICATION",
            transitions=transitions,
            verification_subject=copy.deepcopy(subject),
        )
    triggers = record.get("review_triggers", [])
    if not isinstance(triggers, list) or not all(isinstance(trigger, str) for trigger in triggers):
        raise WorkflowError("review_triggers must be a JSON array of strings")
    requirement = review_required(task, triggers)
    review = _review_for_task(state, key)
    if requirement["required"]:
        if review is None:
            review = {
                "task_key": key,
                "status": "REQUIRED",
                "reviewers": 1,
                "repair_cycles": 0,
                "subject": copy.deepcopy(subject),
                "route": "terra-high",
                "write_ownership": "none",
                "rubric_ref": "docs/workflow/policies/review-matrix.yaml",
            }
            state["reviews"].append(review)
            return _advance_result(state, advanced=False, next_step="AWAIT_COLD_REVIEW", transitions=transitions, review_packet=copy.deepcopy(review))
        if review["status"] == "REPAIR_LIMIT_REACHED":
            return _advance_result(state, advanced=False, next_step="REPAIR_LIMIT_REACHED", transitions=transitions)
        if review["status"] == "REPAIR_IN_PROGRESS" and review["subject"] != subject:
            review["subject"] = copy.deepcopy(subject)
            review["status"] = "REQUIRED"
        review_status = record.get("review_status")
        if review_status is not None:
            if review_status == "PASS":
                review["status"] = "PASS"
            elif review_status == "REPAIR_REQUIRED":
                if review["repair_cycles"] >= 1:
                    review["status"] = "REPAIR_LIMIT_REACHED"
                    return _advance_result(state, advanced=False, next_step="REPAIR_LIMIT_REACHED", transitions=transitions)
                review["repair_cycles"] = 1
                review["status"] = "REPAIR_IN_PROGRESS"
                transition(state, "PLAN")
                transitions.append("PLAN")
                return _advance_result(state, advanced=True, next_step="REPAIR_CYCLE_PERMITTED", transitions=transitions, review_packet=copy.deepcopy(review))
            else:
                raise WorkflowError("review_status must be PASS or REPAIR_REQUIRED")
        if review["status"] != "PASS":
            return _advance_result(state, advanced=False, next_step="AWAIT_COLD_REVIEW", transitions=transitions, review_packet=copy.deepcopy(review))
    transition(state, "INTEGRATE")
    transitions.append("INTEGRATE")
    return _advance_result(
        state,
        advanced=True,
        next_step="AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION",
        transitions=transitions,
        integration_evidence=_integration_evidence(key, handoff, reusable_verification, review),
    )


def load_state(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    return validate_state(state)


def write_state(path: str, state: dict[str, Any]) -> None:
    validate_state(state)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temporary = tempfile.mkstemp(prefix=".temper-workflow-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _record_argument(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        return _as_object(json.loads(value), "--record")
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"--record must be JSON: {exc.msg}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temper-workflow", description=__doc__)
    parser.add_argument("--state", required=True, help="JSON coordinator record")
    parser.add_argument("--record", help="JSON object for the selected command")
    parser.add_argument("command", choices=["status", "validate", "transition", "authorize", "validate-task", "register-worker", "reconcile-worker", "acquire-ownership", "release-ownership", "record-evidence", "record-verification", "verification-reuse", "checkpoint", "compile-context", "ingest-handoff", "review-required", "next-action", "advance"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = load_state(args.state)
        record = _record_argument(args.record)
        changed = False
        if args.command == "status":
            result = {"phase": state["phase"], "authorization_state": state["authorization_state"], "next_action": next_action(state)}
        elif args.command == "validate":
            result = {"valid": True}
        elif args.command == "transition":
            result, changed = transition(state, record.get("target")), True
        elif args.command == "authorize":
            result, changed = authorize(state, record), True
        elif args.command == "validate-task":
            result = {"valid": True, "task": validate_task(record, exact_base=state["repository"]["exact_base"])}
        elif args.command == "register-worker":
            result, changed = register_worker(state, record), True
        elif args.command == "reconcile-worker":
            result, changed = reconcile_worker(state, record), True
        elif args.command == "acquire-ownership":
            result, changed = acquire_ownership(state, record), True
        elif args.command == "release-ownership":
            result, changed = release_ownership(state, record), True
        elif args.command == "record-evidence":
            result, changed = record_evidence(state, record), True
        elif args.command == "record-verification":
            result, changed = record_verification(state, record), True
        elif args.command == "verification-reuse":
            result = verification_reuse(record.get("verification"), record.get("subject"), record.get("requested"))
        elif args.command == "checkpoint":
            result, changed = checkpoint(state, record), True
        elif args.command == "compile-context":
            result = compile_context(state)
        elif args.command == "ingest-handoff":
            result, changed = ingest_handoff(state, record), True
        elif args.command == "review-required":
            result = review_required(record.get("task"), record.get("triggers", []))
        elif args.command == "next-action":
            result = {"next_action": next_action(state)}
        else:
            previous_state = copy.deepcopy(state)
            result = advance(state, record)
            changed = state != previous_state
        if changed:
            write_state(args.state, state)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, WorkflowError) as exc:
        print(f"temper-workflow: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
