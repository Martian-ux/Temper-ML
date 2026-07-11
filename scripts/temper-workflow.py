#!/usr/bin/env python3
"""Fail-closed, JSON-only coordinator for bounded Temper ML workflow tasks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
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
HEARTBEAT_DEFAULT_SECONDS = 120
HEARTBEAT_MAX_SECONDS = 900
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
    "cold_technical_review": ("terra-high", {"high"}),
}
TASK_WRITE_CAPABILITY = {
    "routine_administration": False,
    "mechanical_change": True,
    "normal_implementation": True,
    "protected_boundary_implementation": True,
    "cold_technical_review": False,
}
AUTHORIZATION_PHASES = {
    "PROPOSED": {"RECONCILE"},
    "REVIEWED_WITH_CORRECTIONS": {"RECONCILE"},
    "MAINTAINER_AUTHORIZED": {"RECONCILE", "DELIBERATE", "DECIDE", "PLAN"},
    "IMPLEMENTATION_READY": {"PLAN", "DISPATCH"},
    "IMPLEMENTING": {"DISPATCH", "VERIFY"},
    "VERIFIED": {"VERIFY"},
    "INTEGRATION_AUTHORIZED": {"INTEGRATE"},
    "INTEGRATED": {"CLOSE"},
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
        "event_sequence": 0,
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
    # PurePosixPath treats drive-relative forms such as C:work as relative.
    # Reject all Windows absolute/rooted namespaces before normalizing slashes.
    if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
        raise WorkflowError(f"unsafe owned path: {value!r}")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path == ".":
        raise WorkflowError(f"unsafe owned path: {value!r}")
    return path.as_posix()


def paths_overlap(first: str, second: str) -> bool:
    """Return whether two normalized paths are equal or share a file-tree boundary."""

    return first == second or first.startswith(f"{second}/") or second.startswith(f"{first}/")


def _identity(value: Any, exact_base: str) -> dict[str, str]:
    value = _as_object(value, "identity")
    kind = value.get("type")
    base = value.get("base")
    if kind not in {"git_commit", "patch"}:
        raise WorkflowError("identity type must be git_commit or patch")
    if base != exact_base:
        raise WorkflowError("identity base does not match the authoritative base")
    subject = "head" if kind == "git_commit" else "patch"
    if not isinstance(value.get(subject), str) or not value[subject]:
        raise WorkflowError(f"{kind} identity requires a non-empty {subject}")
    return {"type": kind, "base": base, subject: value[subject]}


def _verification_request(record: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    required = ("command", "scope", "environment", "side_effects")
    if any(name not in record for name in required):
        raise WorkflowError("verification request identity is incomplete")
    return tuple(copy.deepcopy(record[name]) for name in required)


def _next_sequence(state: dict[str, Any]) -> int:
    state["event_sequence"] += 1
    return state["event_sequence"]


def _new_monitor(status: str) -> dict[str, Any]:
    return {
        "interval_seconds": HEARTBEAT_DEFAULT_SECONDS,
        "max_interval_seconds": HEARTBEAT_MAX_SECONDS,
        "unchanged_heartbeats": 0,
        "last_status": status,
        "last_output_identity": None,
        "last_repository_evidence": None,
        "last_heartbeat_at": None,
        "next_check_at": None,
    }


def _validate_monitor(value: Any, status: str) -> dict[str, Any]:
    monitor = _as_object(value, "worker monitor")
    if (
        not isinstance(monitor.get("interval_seconds"), int)
        or not isinstance(monitor.get("max_interval_seconds"), int)
        or monitor["interval_seconds"] < HEARTBEAT_DEFAULT_SECONDS
        or monitor["max_interval_seconds"] < monitor["interval_seconds"]
        or monitor["max_interval_seconds"] > HEARTBEAT_MAX_SECONDS
        or not isinstance(monitor.get("unchanged_heartbeats"), int)
        or monitor["unchanged_heartbeats"] < 0
        or monitor.get("last_status") != status
    ):
        raise WorkflowError("worker monitor metadata is invalid")
    for name in ("last_heartbeat_at", "next_check_at"):
        if monitor.get(name) is not None and (not isinstance(monitor[name], int) or monitor[name] < 0):
            raise WorkflowError("worker monitor timestamps must be non-negative integers")
    return monitor


def _reset_monitor(monitor: dict[str, Any], status: str) -> None:
    monitor.update(_new_monitor(status))


def _routes_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return all(first.get(field, "UNVERIFIED") == second.get(field, "UNVERIFIED") for field in ("selected_model", "selected_effort", "observed_model", "observed_effort"))


def _has_authoritative_decision(
    evidence: dict[str, Any], task_key: str, identity: dict[str, str], exact_base: str, authority_reference: str
) -> bool:
    if evidence.get("kind") != "decision" or evidence.get("task_key") != task_key or evidence.get("subject") != identity:
        return False
    try:
        provenance = _as_object(evidence.get("authoritative_provenance"), "decision authoritative_provenance")
        return (
            provenance.get("task_key") == task_key
            and _identity(provenance.get("subject"), exact_base) == identity
            and provenance.get("authority_reference") == authority_reference
        )
    except WorkflowError:
        return False


def _validate_verifier_registration(record: dict[str, Any], tasks: dict[str, Any], exact_base: str) -> None:
    key = record.get("task_key")
    if key not in tasks:
        raise WorkflowError("verifier registration must bind a registered task")
    _identity(record.get("subject"), exact_base)
    _verification_request(record)
    if not isinstance(record.get("verifier_reference"), str) or not record["verifier_reference"]:
        raise WorkflowError("verifier registration requires a verifier_reference")
    if record.get("accepted") is not True:
        raise WorkflowError("verifier registration must be explicitly accepted")


def _structured_references(value: Any, label: str, task_key: str, identity: dict[str, str]) -> list[dict[str, Any]]:
    references = _as_list(value, label)
    if not references:
        raise WorkflowError(f"{label} must not be empty")
    result = []
    for reference in references:
        reference = _as_object(reference, label)
        if reference.get("task_key") != task_key:
            raise WorkflowError(f"{label} must bind the handoff task")
        if not isinstance(reference.get("reference"), str) or not reference["reference"]:
            raise WorkflowError(f"{label} reference is required")
        if _identity(reference.get("identity"), identity["base"]) != identity:
            raise WorkflowError(f"{label} identity does not match the handoff")
        result.append(reference)
    return result


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
    if route.get("observed_model", "UNVERIFIED") not in {"UNVERIFIED", selected_model}:
        raise WorkflowError("observed_model does not match the selected model")
    if route.get("observed_effort", "UNVERIFIED") not in {"UNVERIFIED", selected_effort}:
        raise WorkflowError("observed_effort does not match the selected reasoning effort")
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
    triggers = _as_list(task.get("review_triggers", []), "task review_triggers")
    if not all(isinstance(trigger, str) and trigger for trigger in triggers) or len(triggers) != len(set(triggers)):
        raise WorkflowError("task review_triggers must be unique non-empty strings")
    authorization = _as_object(task.get("maintainer_authorization"), "task maintainer_authorization")
    identity = _identity(task.get("subject"), task["exact_base"])
    if (
        authorization.get("actor") != "maintainer"
        or authorization.get("task_key") != key
        or authorization.get("exact_base") != task["exact_base"]
        or _identity(authorization.get("subject"), task["exact_base"]) != identity
        or _identity(authorization.get("task_packet_identity"), task["exact_base"]) != identity
    ):
        raise WorkflowError("task maintainer authorization is not bound to the task packet and subject")
    if not all(isinstance(authorization.get(name), str) and authorization[name] for name in ("authority_reference", "readiness_evidence", "reviewed_corrections_evidence")):
        raise WorkflowError("task maintainer authorization provenance is incomplete")
    result = copy.deepcopy(task)
    result["owned_paths"] = owned_paths
    result["review_triggers"] = triggers
    return result


def validate_state(state: Any) -> dict[str, Any]:
    state = _as_object(state, "state")
    if state.get("schema_version") != 1:
        raise WorkflowError("unsupported or missing schema_version")
    if state.get("phase") not in PHASES:
        raise WorkflowError("unknown workflow phase")
    authorization_state = state.get("authorization_state")
    if authorization_state not in AUTHORIZATION_PHASES:
        raise WorkflowError("unknown authorization state")
    if state["phase"] not in AUTHORIZATION_PHASES[authorization_state]:
        raise WorkflowError("authorization state is incompatible with workflow phase")
    mission = _as_object(state.get("mission"), "mission")
    if not isinstance(mission.get("priority"), str) or not mission["priority"]:
        raise WorkflowError("active mission priority is required")
    if mission.get("mission_fit") != "FIT":
        raise WorkflowError("active mission fit must be FIT")
    repository = _as_object(state.get("repository"), "repository")
    if not isinstance(repository.get("exact_base"), str) or not repository["exact_base"]:
        raise WorkflowError("authoritative exact base is required")
    if not isinstance(state.get("event_sequence"), int) or state["event_sequence"] < 0:
        raise WorkflowError("event_sequence must be a non-negative integer")
    tasks = _as_object(state.get("tasks"), "tasks")
    for key, task in tasks.items():
        if key != validate_task(task, exact_base=repository["exact_base"])["task_key"]:
            raise WorkflowError("task registry key does not match task_key")
    ownership = _as_list(state.get("ownership"), "ownership")
    active_paths: list[str] = []
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
        overlap = [(existing, path) for existing in active_paths for path in paths if paths_overlap(existing, path)]
        if overlap:
            raise WorkflowError(f"active ownership path collision: {overlap!r}")
        active_paths.extend(paths)
    workers = _as_list(state.get("workers"), "workers")
    seen_worker_refs: set[str] = set()
    active_worker_tasks: set[str] = set()
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
        attempt = worker.get("attempt", 1)
        if not isinstance(attempt, int) or attempt < 1:
            raise WorkflowError("worker attempt must be a positive integer")
        if worker.get("status") == "SPAWN_UNKNOWN" and not isinstance(worker.get("ambiguity_reference"), str):
            raise WorkflowError("ambiguous worker state requires durable ambiguity_reference")
        if worker.get("status") in ACTIVE_WORKER_STATES:
            key = worker.get("task_key")
            task = tasks.get(key)
            if task is None:
                raise WorkflowError("active worker has no registered matching task")
            if worker.get("task_class") != task["task_class"]:
                raise WorkflowError("active worker task class does not match its task")
            validate_route(worker.get("route"), worker["task_class"])
            if not _routes_match(worker["route"], task["route"]):
                raise WorkflowError("active worker route does not match its task")
            derived_writer = TASK_WRITE_CAPABILITY[task["task_class"]]
            if worker.get("writer") is not derived_writer:
                raise WorkflowError("worker write capability does not match its task class")
            if derived_writer and key not in active_keys:
                raise WorkflowError("active writer has no active ownership lease")
            if key in active_worker_tasks:
                raise WorkflowError("more than one active or spawn-requested worker exists for a task")
            active_worker_tasks.add(key)
        _validate_monitor(worker.get("monitor"), worker["status"])
    if _active_writer_count(state) > 2:
        raise WorkflowError("a third active implementation writer is prohibited")
    for field in ("evidence", "verification", "checkpoints", "handoffs", "reviews"):
        _as_list(state.get(field), field)
    for evidence in state["evidence"]:
        evidence = _as_object(evidence, "evidence")
        if evidence.get("kind") == "decision":
            key = evidence.get("task_key")
            if key not in tasks or not _has_authoritative_decision(
                evidence,
                key,
                _identity(evidence.get("subject"), repository["exact_base"]),
                repository["exact_base"],
                tasks[key]["maintainer_authorization"]["authority_reference"],
            ):
                raise WorkflowError("decision evidence lacks authoritative task/subject provenance")
        elif evidence.get("kind") == "verifier_registration":
            _validate_verifier_registration(evidence, tasks, repository["exact_base"])
    for verification in state["verification"]:
        verification = _as_object(verification, "verification record")
        key = verification.get("task_key")
        if key not in tasks:
            raise WorkflowError("verification must bind a registered task_key")
        _verification_request(verification)
        _identity(verification.get("subject"), repository["exact_base"])
        if verification.get("status") not in {"PASS", "FAIL", "UNKNOWN", "NON_REUSABLE"}:
            raise WorkflowError("unknown verification status")
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
    if target == "INTEGRATE":
        raise WorkflowError("integration transition requires maintainer integration authorization")
    if target not in AUTHORIZATION_PHASES[state["authorization_state"]]:
        raise WorkflowError("authorization state is incompatible with workflow phase")
    state["phase"] = target
    return {"phase": target}


def _transition_with_authorization(state: dict[str, Any], target: str, authorization_state: str) -> None:
    """Apply a coordinator-controlled phase and lifecycle update atomically."""

    validate_state(state)
    if target not in ALLOWED_TRANSITIONS[state["phase"]]:
        raise WorkflowError(f"invalid transition {state['phase']} -> {target}")
    state["phase"] = target
    state["authorization_state"] = authorization_state
    validate_state(state)


def authorize(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if record.get("actor") != "maintainer":
        raise WorkflowError("only a maintainer may authorize")
    target = record.get("authorization_state", "MAINTAINER_AUTHORIZED")
    if target not in {"MAINTAINER_AUTHORIZED", "INTEGRATION_AUTHORIZED"}:
        raise WorkflowError("only maintainer authorization states may be granted")
    if target == "MAINTAINER_AUTHORIZED":
        if state["phase"] != "RECONCILE" or state["authorization_state"] not in {"PROPOSED", "REVIEWED_WITH_CORRECTIONS", "MAINTAINER_AUTHORIZED"}:
            raise WorkflowError("maintainer authorization is only valid from reconciliation")
    else:
        _authorize_integration(state, record)
    state["authorization_state"] = target
    if target == "INTEGRATION_AUTHORIZED":
        state["phase"] = "INTEGRATE"
    return {"authorization_state": target, "actor": "maintainer", "phase": state["phase"]}


def register_task(state: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    task = validate_task(task, exact_base=state["repository"]["exact_base"])
    if state["authorization_state"] != "MAINTAINER_AUTHORIZED":
        raise WorkflowError("authoritative state is not maintainer-authorized")
    if state["phase"] != "RECONCILE":
        raise WorkflowError("task registration is only valid in RECONCILE")
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
    if state["authorization_state"] not in {"MAINTAINER_AUTHORIZED", "IMPLEMENTATION_READY"}:
        raise WorkflowError("authoritative state is not currently ready for implementation ownership")
    if state["phase"] != "PLAN":
        raise WorkflowError("ownership acquisition is only valid in PLAN")
    key = record.get("task_key")
    task = state["tasks"].get(key)
    if task is None:
        raise WorkflowError("ownership requires a registered task")
    if not TASK_WRITE_CAPABILITY[task["task_class"]]:
        raise WorkflowError("read-only task class cannot acquire WRITE ownership")
    paths = [normalized_path(path) for path in _as_list(record.get("paths"), "paths")]
    if set(paths) != set(task["owned_paths"]):
        raise WorkflowError("ownership paths must exactly match the task paths")
    for lease in state["ownership"]:
        if lease.get("active") and (
            lease.get("task_key") == key
            or any(paths_overlap(normalized_path(existing), path) for existing in lease["paths"] for path in paths)
        ):
            raise WorkflowError("task-key or path ownership collision")
    lease = {"task_key": key, "paths": paths, "active": True}
    state["ownership"].append(lease)
    state["authorization_state"] = "IMPLEMENTATION_READY"
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
    count = _active_writer_count(state)
    if count >= 2:
        raise WorkflowError("a third active implementation writer is prohibited")
    if count == 1:
        guards = record.get("independence_guards")
        if not isinstance(guards, dict) or not all(guards.get(name) is True for name in REQUIRED_SECOND_WRITER_GUARDS):
            raise WorkflowError("second writer requires every recorded independence guard")


def register_worker(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if state["authorization_state"] != "IMPLEMENTING" or state["phase"] != "DISPATCH":
        raise WorkflowError("worker registration is only valid for an implementing dispatch")
    key = record.get("task_key")
    if key not in state["tasks"]:
        raise WorkflowError("worker requires a registered task")
    task = state["tasks"][key]
    writer = TASK_WRITE_CAPABILITY[task["task_class"]]
    if "writer" in record and record["writer"] is not writer:
        raise WorkflowError("worker write capability is derived from task class")
    supplied_route = record.get("route")
    if supplied_route is not None:
        validate_route(supplied_route, task["task_class"])
        if not _routes_match(supplied_route, task["route"]):
            raise WorkflowError("worker supplied route does not match its task")
    if writer and not any(lease.get("active") and lease.get("task_key") == key for lease in state["ownership"]):
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
    if manual_intent is None:
        raise WorkflowError("worker registration requires a durable task-specific SPAWN_REQUESTED intent")
    if any(worker is not manual_intent and worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES for worker in state["workers"]):
        raise WorkflowError("task already has an active worker")
    manual_intent["reference"] = reference
    manual_intent["status"] = "ACTIVE"
    _reset_monitor(manual_intent["monitor"], "ACTIVE")
    return manual_intent


def reconcile_worker(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    reference = record.get("reference")
    for worker in state["workers"]:
        if worker.get("reference") != reference:
            continue
        target = record.get("status")
        if target is not None:
            allowed = {
                "SPAWN_REQUESTED": {"ACTIVE", "SPAWN_UNKNOWN", "RETIRED"},
                "ACTIVE": {"WAITING", "BLOCKED", "COMPLETE", "RETIRED"},
                "WAITING": {"ACTIVE", "BLOCKED", "COMPLETE", "RETIRED"},
                "BLOCKED": {"ACTIVE", "WAITING", "RETIRED"},
                "SPAWN_UNKNOWN": {"RETIRED"},
            }
            if target not in allowed.get(worker.get("status"), set()):
                raise WorkflowError("illegal worker attempt-state transition")
            if worker.get("status") == "SPAWN_UNKNOWN" and record.get("maintainer_replacement_decision") is not True:
                raise WorkflowError("SPAWN_UNKNOWN requires a maintainer replacement decision")
            if target == "SPAWN_UNKNOWN":
                if not isinstance(record.get("ambiguity_reference"), str) or not record["ambiguity_reference"]:
                    raise WorkflowError("SPAWN_UNKNOWN requires durable ambiguity_reference")
                worker["ambiguity_reference"] = record["ambiguity_reference"]
            worker["status"] = target
            _reset_monitor(worker["monitor"], target)
            return {"reference": reference, "status": target}
        if worker.get("status") == "SPAWN_UNKNOWN":
            if record.get("maintainer_replacement_decision") is not True:
                return {"reference": reference, "status": "SPAWN_UNKNOWN", "retry": "BLOCKED"}
            worker["status"] = "RETIRED"
            _reset_monitor(worker["monitor"], "RETIRED")
            return {"reference": reference, "status": "RETIRED", "replacement": "MAINTAINER_APPROVED"}
        return {"reference": reference, "status": worker["status"], "retry": "NOT_NEEDED"}
    raise WorkflowError("unknown worker reference")


def monitor_worker(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Persist deterministic, low-frequency polling guidance for one worker.

    This is deliberately a record operation, not a timer or background service.
    Callers provide an integer clock value and may wait longer than the returned
    interval without changing the result's safety properties.
    """

    validate_state(state)
    reference = record.get("reference")
    worker = next((item for item in state["workers"] if item.get("reference") == reference), None)
    if worker is None:
        raise WorkflowError("unknown worker reference")
    now = record.get("now")
    if not isinstance(now, int) or now < 0:
        raise WorkflowError("worker monitor requires a non-negative integer now")
    monitor = _validate_monitor(worker["monitor"], worker["status"])
    if record.get("system_error") is True:
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now
        return {"action": "WAKE_IMMEDIATELY", "reason": "SYSTEM_ERROR", "reference": reference}
    if record.get("maintainer_input") is True:
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now
        return {"action": "WAKE_IMMEDIATELY", "reason": "MAINTAINER_INPUT", "reference": reference}
    observed_status = record.get("status", worker["status"])
    if observed_status not in ACTIVE_WORKER_STATES | {"COMPLETE", "RETIRED"}:
        raise WorkflowError("unknown monitored worker status")
    if worker["status"] == "SPAWN_UNKNOWN":
        return {"action": "WAKE_IMMEDIATELY", "reason": "AMBIGUOUS_SPAWN_RECONCILIATION", "reference": reference}
    if observed_status != worker["status"]:
        allowed = {
            "SPAWN_REQUESTED": {"ACTIVE", "SPAWN_UNKNOWN", "RETIRED"},
            "ACTIVE": {"WAITING", "BLOCKED", "COMPLETE", "RETIRED"},
            "WAITING": {"ACTIVE", "BLOCKED", "COMPLETE", "RETIRED"},
            "BLOCKED": {"ACTIVE", "WAITING", "RETIRED"},
        }
        if observed_status not in allowed.get(worker["status"], set()):
            raise WorkflowError("illegal monitored worker attempt-state transition")
        if observed_status == "SPAWN_UNKNOWN":
            ambiguity = record.get("ambiguity_reference")
            if not isinstance(ambiguity, str) or not ambiguity:
                raise WorkflowError("SPAWN_UNKNOWN requires durable ambiguity_reference")
            worker["ambiguity_reference"] = ambiguity
        worker["status"] = observed_status
        _reset_monitor(monitor, observed_status)
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now if observed_status in {"COMPLETE", "SPAWN_UNKNOWN"} else now + HEARTBEAT_DEFAULT_SECONDS
        if observed_status in {"COMPLETE", "SPAWN_UNKNOWN", "RETIRED"}:
            reason = "COMPLETION" if observed_status == "COMPLETE" else "AMBIGUOUS_SPAWN_RECONCILIATION" if observed_status == "SPAWN_UNKNOWN" else "TERMINAL_STATUS"
            return {"action": "WAKE_IMMEDIATELY", "reason": reason, "reference": reference}
        return {"action": "HEARTBEAT_CHANGED", "reference": reference, "interval_seconds": monitor["interval_seconds"], "next_check_at": monitor["next_check_at"]}
    if worker["status"] in {"COMPLETE", "RETIRED"}:
        return {"action": "WAKE_IMMEDIATELY", "reason": "TERMINAL_STATUS", "reference": reference}
    for name in ("output_identity", "repository_evidence"):
        if name in record and record[name] is not None and (not isinstance(record[name], str) or not record[name]):
            raise WorkflowError(f"worker monitor {name} must be a non-empty string when supplied")
    output_identity = record.get("output_identity", monitor["last_output_identity"])
    repository_evidence = record.get("repository_evidence", monitor["last_repository_evidence"])
    changed = output_identity != monitor["last_output_identity"] or repository_evidence != monitor["last_repository_evidence"]
    if changed:
        _reset_monitor(monitor, worker["status"])
        monitor["last_output_identity"] = output_identity
        monitor["last_repository_evidence"] = repository_evidence
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now + HEARTBEAT_DEFAULT_SECONDS
        return {"action": "HEARTBEAT_CHANGED", "reference": reference, "interval_seconds": monitor["interval_seconds"], "next_check_at": monitor["next_check_at"]}
    if monitor["last_heartbeat_at"] is None:
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now + monitor["interval_seconds"]
    elif now >= monitor["next_check_at"]:
        monitor["interval_seconds"] = min(monitor["interval_seconds"] * 2, monitor["max_interval_seconds"])
        monitor["unchanged_heartbeats"] += 1
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now + monitor["interval_seconds"]
    return {
        "action": "WAIT_UNTIL_HEARTBEAT",
        "reference": reference,
        "interval_seconds": monitor["interval_seconds"],
        "next_check_at": monitor["next_check_at"],
    }


def record_evidence(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if not isinstance(record.get("kind"), str) or not record["kind"]:
        raise WorkflowError("evidence kind is required")
    if not isinstance(record.get("subject"), dict):
        raise WorkflowError("evidence subject is required")
    if record["kind"] == "decision":
        key = record.get("task_key")
        if key not in state["tasks"]:
            raise WorkflowError("decision evidence must bind a registered task")
        subject = _identity(record["subject"], state["repository"]["exact_base"])
        provenance = _as_object(record.get("authoritative_provenance"), "decision authoritative_provenance")
        accepted_authority = state["tasks"][key]["maintainer_authorization"]["authority_reference"]
        if (
            provenance.get("task_key") != key
            or _identity(provenance.get("subject"), state["repository"]["exact_base"]) != subject
            or provenance.get("authority_reference") != accepted_authority
        ):
            raise WorkflowError("decision evidence lacks authoritative task/subject provenance")
    elif record["kind"] == "verifier_registration":
        _validate_verifier_registration(record, state["tasks"], state["repository"]["exact_base"])
    stored = copy.deepcopy(record)
    stored["sequence"] = _next_sequence(state)
    state["evidence"].append(stored)
    return {"recorded": "evidence", "count": len(state["evidence"])}


def record_verification(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if state["phase"] != "VERIFY" or state["authorization_state"] not in {"IMPLEMENTING", "VERIFIED"}:
        raise WorkflowError("verification recording is only valid in VERIFY")
    key = record.get("task_key")
    if key not in state["tasks"]:
        raise WorkflowError("verification requires a registered task_key")
    for name in ("reference", "subject", "command", "scope", "environment", "side_effects", "status"):
        if name not in record:
            raise WorkflowError(f"verification {name} is required")
    if not isinstance(record["reference"], str) or not record["reference"]:
        raise WorkflowError("verification reference is required")
    if record["status"] not in {"PASS", "FAIL", "UNKNOWN", "NON_REUSABLE"}:
        raise WorkflowError("unknown verification status")
    subject = _identity(record["subject"], state["repository"]["exact_base"])
    request = _verification_request(record)
    for existing in state["verification"]:
        if existing.get("reference") == record["reference"] and not (
            existing.get("task_key") == key
            and _identity(existing.get("subject"), state["repository"]["exact_base"]) == subject
            and _verification_request(existing) == request
        ):
            raise WorkflowError("verification reference must be unique to task, subject, and request identity")
    stored = copy.deepcopy(record)
    stored["subject"] = subject
    stored["sequence"] = _next_sequence(state)
    state["verification"].append(stored)
    _refresh_verification_authorization(state)
    return {"recorded": "verification", "count": len(state["verification"])}


def verification_reuse(record: dict[str, Any], subject: dict[str, Any], requested: Any = None) -> dict[str, Any]:
    record = _as_object(record, "verification record")
    subject = _as_object(subject, "subject")
    required = ("task_key", "subject", "command", "scope", "environment", "side_effects", "status")
    if any(name not in record for name in required) or record.get("status") != "PASS":
        return {"reusable": False, "status": "NON_REUSABLE"}
    try:
        exact_base = subject.get("base")
        if not isinstance(exact_base, str) or _identity(subject, exact_base) != _identity(record["subject"], exact_base):
            return {"reusable": False, "status": "NON_REUSABLE"}
    except WorkflowError:
        return {"reusable": False, "status": "NON_REUSABLE"}
    if record.get("invalidated"):
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
    if state["phase"] != "DISPATCH" or state["authorization_state"] != "IMPLEMENTING":
        raise WorkflowError("handoff ingestion is only valid in an implementing dispatch")
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
    for name in ("identity", "route", "acceptance_evidence", "verification", "applied_decisions", "verification_references", "scope_safety", "open_findings", "integration_guidance"):
        if name not in record:
            raise WorkflowError(f"handoff {name} is required")
    identity = _identity(record["identity"], state["repository"]["exact_base"])
    applied_decisions = _structured_references(record["applied_decisions"], "handoff applied_decisions", key, identity)
    verification_references = _structured_references(record["verification_references"], "handoff verification_references", key, identity)
    for reference in verification_references:
        if "verification_request" not in reference:
            raise WorkflowError("handoff verification_references must bind a request identity")
        _verification_request(reference["verification_request"])
    for decision in applied_decisions:
        if not any(
            _has_authoritative_decision(
                evidence,
                key,
                identity,
                state["repository"]["exact_base"],
                task["maintainer_authorization"]["authority_reference"],
            )
            and evidence.get("reference") == decision["reference"]
            for evidence in state["evidence"]
        ):
            raise WorkflowError("handoff applied decision is not an authoritative task decision")
    if not _as_list(record["acceptance_evidence"], "handoff acceptance_evidence"):
        raise WorkflowError("handoff acceptance_evidence must not be empty")
    if not _as_list(record["verification"], "handoff verification"):
        raise WorkflowError("handoff verification must not be empty")
    validate_route(record["route"], task["task_class"])
    if not _routes_match(record["route"], task["route"]):
        raise WorkflowError("handoff selected route does not match its task")
    worker["status"] = "COMPLETE"
    _reset_monitor(worker["monitor"], "COMPLETE")
    stored = copy.deepcopy(record)
    stored["identity"] = identity
    stored["applied_decisions"] = applied_decisions
    stored["verification_references"] = verification_references
    stored["sequence"] = _next_sequence(state)
    state["handoffs"].append(stored)
    if state["phase"] == "DISPATCH":
        transition(state, "VERIFY")
    return {"task_key": key, "worker_reference": worker_reference, "ingested": True}


def review_required(task: dict[str, Any], triggers: list[str] | None = None) -> dict[str, Any]:
    task = validate_task(task)
    if task["task_class"] == "protected_boundary_implementation":
        return {"required": True, "reviewers": 1, "route": "terra-high", "reason": "protected_boundary"}
    stored_triggers = task["review_triggers"]
    return {
        "required": bool(stored_triggers),
        "reviewers": 1 if stored_triggers else 0,
        "route": "terra-high" if stored_triggers else None,
        "reason": "risk_trigger" if stored_triggers else "no_matrix_trigger",
        "triggers": copy.deepcopy(stored_triggers),
    }


def next_action(state: dict[str, Any]) -> str:
    if state["authorization_state"] in {"PROPOSED", "REVIEWED_WITH_CORRECTIONS"}:
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
    state: dict[str, Any], task_key: str, subject: dict[str, Any], requested: Any
) -> dict[str, Any] | None:
    verification = _latest_verification(state, task_key, subject, requested)
    if verification is None:
        return None
    return verification if verification_reuse(verification, subject, requested)["reusable"] else None


def _latest_verification(
    state: dict[str, Any], task_key: str, subject: dict[str, Any], requested: Any
) -> dict[str, Any] | None:
    try:
        request_identity = _verification_request(_as_object(requested, "verification request"))
    except WorkflowError:
        return None
    for verification in reversed(state["verification"]):
        if verification.get("task_key") != task_key:
            continue
        try:
            if (
                _identity(verification.get("subject"), subject["base"]) == subject
                and _verification_request(verification) == request_identity
            ):
                return verification
        except WorkflowError:
            return None
    return None


def _refresh_verification_authorization(state: dict[str, Any]) -> None:
    if state["phase"] != "VERIFY" or state["authorization_state"] not in {"IMPLEMENTING", "VERIFIED"}:
        return
    seen_tasks: set[str] = set()
    for handoff in reversed(state["handoffs"]):
        key = handoff.get("task_key")
        if key in seen_tasks:
            continue
        seen_tasks.add(key)
        subject = handoff.get("identity")
        for reference in handoff.get("verification_references", []):
            requested = reference.get("verification_request")
            latest = _latest_verification(state, key, subject, requested)
            if latest is not None and verification_reuse(latest, subject, requested)["reusable"] and _matching_handoff_verification(handoff, latest):
                state["authorization_state"] = "VERIFIED"
                return
    state["authorization_state"] = "IMPLEMENTING"


def _registered_verifier(
    state: dict[str, Any], task_key: str, subject: dict[str, Any], requested: dict[str, Any], verifier_reference: Any
) -> bool:
    if not isinstance(verifier_reference, str) or not verifier_reference:
        return False
    request_identity = _verification_request(requested)
    return any(
        evidence.get("kind") == "verifier_registration"
        and evidence.get("task_key") == task_key
        and evidence.get("subject") == subject
        and evidence.get("accepted") is True
        and evidence.get("verifier_reference") == verifier_reference
        and _verification_request(evidence) == request_identity
        for evidence in state["evidence"]
    )


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


def _matching_handoff_verification(handoff: dict[str, Any], verification: dict[str, Any]) -> bool:
    reference = verification.get("reference")
    return any(
        item.get("reference") == reference
        and item.get("task_key") == handoff["task_key"]
        and item.get("identity") == handoff["identity"]
        and verification.get("task_key") == handoff["task_key"]
        and verification.get("subject") == handoff["identity"]
        and _verification_request(item.get("verification_request", {})) == _verification_request(verification)
        for item in handoff.get("verification_references", [])
    )


def _authorize_integration(state: dict[str, Any], record: dict[str, Any]) -> None:
    if state["phase"] != "VERIFY" or state["authorization_state"] != "VERIFIED":
        raise WorkflowError("integration authorization requires a verified candidate in VERIFY")
    key = record.get("task_key")
    if key not in state["tasks"]:
        raise WorkflowError("integration authorization requires a registered task_key")
    handoff = _latest_handoff(state, key)
    if handoff is None:
        raise WorkflowError("integration authorization requires a completed handoff")
    identity = _identity(record.get("identity"), state["repository"]["exact_base"])
    if identity != handoff["identity"]:
        raise WorkflowError("integration authorization identity does not match the handoff")
    decision = _as_object(record.get("applied_decision"), "applied_decision")
    if decision not in handoff["applied_decisions"]:
        raise WorkflowError("integration authorization requires a matching applied decision")
    reference = record.get("verification_reference")
    if not isinstance(reference, str) or not reference:
        raise WorkflowError("integration authorization requires a verification reference")
    requested = record.get("verification_request")
    verification = _reusable_verification(state, key, identity, requested)
    if verification is not None and verification.get("reference") != reference:
        verification = None
    if verification is None or not _matching_handoff_verification(handoff, verification):
        raise WorkflowError("integration authorization requires a handoff-bound verification reference")
    if verification_reuse(verification, identity, requested)["reusable"] is not True:
        raise WorkflowError("integration authorization requires a matching reusable PASS verification")
    review = _review_for_task(state, key)
    if review_required(state["tasks"][key])["required"] and review is None:
        raise WorkflowError("integration authorization requires a passing review of the exact subject")
    if review is not None and (review.get("status") != "PASS" or review.get("subject") != identity):
        raise WorkflowError("integration authorization requires a passing review of the exact subject")
    review_sequence = review.get("completed_sequence", review.get("sequence", 0)) if review is not None else 0
    final_gate = None
    for item in reversed(state["verification"]):
        if item.get("task_key") != key or item.get("subject") != identity or item.get("command") != ["python", "scripts/temper-gate.py", "all"]:
            continue
        final_request = {name: item[name] for name in ("command", "scope", "environment", "side_effects")}
        if (
            _latest_verification(state, key, identity, final_request) is item
            and item.get("status") == "PASS"
            and item.get("sequence", 0) > handoff.get("sequence", 0)
            and item.get("sequence", 0) > review_sequence
            and _registered_verifier(state, key, identity, final_request, item.get("verifier_reference"))
        ):
            final_gate = item
            break
    if final_gate is None:
        raise WorkflowError("integration authorization requires current exact-subject registered-verifier temper-gate all PASS evidence after final assembly")
    public_safety = None
    for item in reversed(state["verification"]):
        if item.get("task_key") != key or item.get("subject") != identity or item.get("verification_type") != "public_safety":
            continue
        public_request = {name: item[name] for name in ("command", "scope", "environment", "side_effects")}
        if _latest_verification(state, key, identity, public_request) is item and item.get("status") == "PASS":
            public_safety = item
            break
    if public_safety is None:
        raise WorkflowError("integration authorization requires exact-subject public-safety PASS evidence")


def _create_dispatch_intent(state: dict[str, Any], task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    key = task["task_key"]
    if any(worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES for worker in state["workers"]):
        raise WorkflowError("task already has an active or spawn-requested worker")
    writer = TASK_WRITE_CAPABILITY[task["task_class"]]
    if not writer:
        raise WorkflowError("read-only task class cannot dispatch a writer")
    if "writer" in record and record["writer"] is not writer:
        raise WorkflowError("worker write capability is derived from task class")
    supplied_route = record.get("route")
    if supplied_route is not None:
        validate_route(supplied_route, task["task_class"])
        if not _routes_match(supplied_route, task["route"]):
            raise WorkflowError("dispatch supplied route does not match its task")
    if not any(lease.get("active") and lease.get("task_key") == key for lease in state["ownership"]):
        raise WorkflowError("dispatch requires an active ownership lease")
    validate_route(task["route"], task["task_class"])
    _check_writer_capacity(state, record)
    intent = {
        "reference": f"manual:{key}",
        "task_key": key,
        "task_class": task["task_class"],
        "writer": writer,
        "status": "SPAWN_REQUESTED",
        "route": copy.deepcopy(task["route"]),
        "attempt": record.get("attempt", 1),
        "independence_guards": copy.deepcopy(record.get("independence_guards")),
        "monitor": _new_monitor("SPAWN_REQUESTED"),
    }
    if any(worker["reference"] == intent["reference"] for worker in state["workers"]):
        raise WorkflowError("manual dispatch intent already exists")
    state["workers"].append(intent)
    return intent


def advance(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if state["phase"] == "INTEGRATE":
        return _advance_result(state, advanced=False, next_step="AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION", transitions=[])
    if state["phase"] == "CLOSE":
        return _advance_result(state, advanced=False, next_step="STOP", transitions=[])
    if state["authorization_state"] not in {"MAINTAINER_AUTHORIZED", "IMPLEMENTATION_READY", "IMPLEMENTING", "VERIFIED"}:
        return _advance_result(state, advanced=False, next_step="STOP_FOR_MAINTAINER_AUTHORIZATION", transitions=[])
    if any(worker.get("status") == "SPAWN_UNKNOWN" for worker in state["workers"]):
        return _advance_result(state, advanced=False, next_step="RECONCILE_AMBIGUOUS_SPAWN", transitions=[])
    key = record.get("task_key")
    task = state["tasks"].get(key)
    if task is None:
        raise WorkflowError("advance requires a registered task_key")
    if state["phase"] == "DISPATCH":
        existing = next((worker for worker in state["workers"] if worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES), None)
        if existing is not None:
            return _advance_result(state, advanced=False, next_step="AWAIT_MANUAL_WORKER_REFERENCE", transitions=[])
        _create_dispatch_intent(state, task, record)
        return _advance_result(
            state,
            advanced=True,
            next_step="AWAIT_MANUAL_WORKER_REFERENCE",
            transitions=[],
            manual_launch_packet={"task_key": key, "owned_paths": task["owned_paths"], "selected_route": task["route"], "manual_adapter": True},
        )
    transitions: list[str] = []
    while state["phase"] in {"RECONCILE", "DELIBERATE", "DECIDE"}:
        target = next(iter(ALLOWED_TRANSITIONS[state["phase"]]))
        transition(state, target)
        transitions.append(target)

    if state["phase"] == "PLAN":
        if state["authorization_state"] != "IMPLEMENTATION_READY":
            return _advance_result(state, advanced=False, next_step="ACQUIRE_REQUIRED_OWNERSHIP", transitions=transitions)
        _create_dispatch_intent(state, task, record)
        transition(state, "DISPATCH")
        state["authorization_state"] = "IMPLEMENTING"
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
    reusable_verification = _reusable_verification(state, key, subject, requested_verification)
    if reusable_verification is None:
        return _advance_result(
            state,
            advanced=False,
            next_step="RUN_OR_RECORD_REQUIRED_VERIFICATION",
            transitions=transitions,
            verification_subject=copy.deepcopy(subject),
        )
    if state["authorization_state"] != "VERIFIED":
        return _advance_result(state, advanced=False, next_step="RUN_OR_RECORD_REQUIRED_VERIFICATION", transitions=transitions)
    if not _matching_handoff_verification(handoff, reusable_verification):
        return _advance_result(state, advanced=False, next_step="RUN_OR_RECORD_REQUIRED_VERIFICATION", transitions=transitions)
    requirement = review_required(task)
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
                "sequence": _next_sequence(state),
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
                review["completed_sequence"] = _next_sequence(state)
            elif review_status == "REPAIR_REQUIRED":
                if review["repair_cycles"] >= 1:
                    review["status"] = "REPAIR_LIMIT_REACHED"
                    return _advance_result(state, advanced=False, next_step="REPAIR_LIMIT_REACHED", transitions=transitions)
                review["repair_cycles"] = 1
                review["status"] = "REPAIR_IN_PROGRESS"
                _transition_with_authorization(state, "PLAN", "MAINTAINER_AUTHORIZED")
                transitions.append("PLAN")
                return _advance_result(state, advanced=True, next_step="REPAIR_CYCLE_PERMITTED", transitions=transitions, review_packet=copy.deepcopy(review))
            else:
                raise WorkflowError("review_status must be PASS or REPAIR_REQUIRED")
        if review["status"] != "PASS":
            return _advance_result(state, advanced=False, next_step="AWAIT_COLD_REVIEW", transitions=transitions, review_packet=copy.deepcopy(review))
    return _advance_result(
        state,
        advanced=False,
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
    parser.add_argument("command", choices=["status", "validate", "transition", "authorize", "validate-task", "register-worker", "reconcile-worker", "monitor-worker", "acquire-ownership", "release-ownership", "record-evidence", "record-verification", "verification-reuse", "checkpoint", "compile-context", "ingest-handoff", "review-required", "next-action", "advance"])
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
        elif args.command == "monitor-worker":
            result, changed = monitor_worker(state, record), True
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
