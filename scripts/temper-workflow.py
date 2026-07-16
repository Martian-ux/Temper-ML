#!/usr/bin/env python3
"""Fail-closed, JSON-only coordinator for bounded Temper ML workflow tasks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
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
ACTIVE_WORKER_STATES = {
    "SPAWN_REQUESTED",
    "SPAWN_UNKNOWN",
    "ACTIVE",
    "WAITING",
    "BLOCKED",
}
HEARTBEAT_DEFAULT_SECONDS = 120
HEARTBEAT_MAX_SECONDS = 900
FINAL_GATE_COMMAND = ["python", "scripts/temper-gate.py", "all"]
ROOT_EXECUTOR = "root"
REQUIRED_SECOND_WRITER_GUARDS = {
    "disjoint_paths",
    "independent_acceptance_criteria",
    "no_uncommitted_output_dependency",
    "no_shared_blocking_decision",
    "known_integration_order",
    "explicit_recorded_approval",
}
PUBLIC_ROUTE = {
    "routine_administration": ("luna-or-cheapest", {"low", "medium"}),
    "mechanical_change": ("terra-medium", {"medium"}),
    "normal_implementation": ("terra-high", {"high"}),
    "protected_boundary_implementation": ("terra-high", {"high"}),
    "cold_technical_review": ("terra-high", {"high"}),
}
ROUTE_MODEL_MARKER = {
    "luna-or-cheapest": "luna",
    "terra-medium": "terra",
    "terra-high": "terra",
    "sol-high": "sol",
    "sol-ultra": "sol",
}
TRIAL_PROTOCOL_REF = "docs/workflow/procedures/model-route-and-experiment.md"
TRIAL_TASK_MIX = {
    "fixed_snapshot_cold_review": 2,
    "bounded_implementation_or_repair": 2,
    "architecture_or_invariant_design": 1,
    "mechanical_negative_control": 1,
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
        "route_trials": [],
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

    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


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
        if monitor.get(name) is not None and (
            not isinstance(monitor[name], int) or monitor[name] < 0
        ):
            raise WorkflowError(
                "worker monitor timestamps must be non-negative integers"
            )
    return monitor


def _reset_monitor(monitor: dict[str, Any], status: str) -> None:
    monitor.update(_new_monitor(status))


def _routes_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return all(
        first.get(field) == second.get(field)
        for field in (
            "declared_route",
            "declared_model",
            "declared_effort",
            "selected_model",
            "selected_effort",
            "selection_mechanism",
            "runtime_observation",
            "declared_route_compliance",
            "experiment",
        )
    )


def _validate_selection_mechanism(value: Any) -> dict[str, Any]:
    mechanism = _as_object(value, "route selection_mechanism")
    kind = mechanism.get("kind")
    required = {
        "host_controls": {
            "model_selector": "model",
            "effort_selector": "reasoning_effort",
            "context_field": "control_surface",
        },
        "cli_flags": {
            "model_selector": "--model",
            "effort_selector": "model_reasoning_effort",
            "context_field": None,
        },
        "cli_profile": {
            "model_selector": "model",
            "effort_selector": "model_reasoning_effort",
            "context_field": "profile",
        },
        "api_parameters": {
            "model_selector": "model",
            "effort_selector": "reasoning.effort",
            "context_field": None,
        },
    }.get(kind)
    if required is None:
        raise WorkflowError(
            "route selection mechanism must be host_controls, cli_flags, "
            "cli_profile, or api_parameters"
        )
    if (
        mechanism.get("model_selector") != required["model_selector"]
        or mechanism.get("effort_selector") != required["effort_selector"]
    ):
        raise WorkflowError(
            "route selection mechanism does not name executable model and effort "
            "selectors"
        )
    context_field = required["context_field"]
    if context_field is not None and (
        not isinstance(mechanism.get(context_field), str)
        or not mechanism[context_field]
    ):
        raise WorkflowError(
            f"route selection mechanism requires non-empty {context_field}"
        )
    return mechanism


def _has_authoritative_decision(
    evidence: dict[str, Any],
    task_key: str,
    identity: dict[str, str],
    exact_base: str,
    authority_reference: str,
) -> bool:
    if (
        evidence.get("kind") != "decision"
        or evidence.get("task_key") != task_key
        or evidence.get("subject") != identity
    ):
        return False
    try:
        provenance = _as_object(
            evidence.get("authoritative_provenance"),
            "decision authoritative_provenance",
        )
        return (
            provenance.get("task_key") == task_key
            and _identity(provenance.get("subject"), exact_base) == identity
            and provenance.get("authority_reference") == authority_reference
        )
    except WorkflowError:
        return False


def _writer_exception_matches(
    evidence: Any,
    task: dict[str, Any],
    exact_base: str,
    writer_mode: str | None = None,
) -> bool:
    try:
        evidence = _as_object(evidence, "writer exception decision")
        identity = _identity(task.get("subject"), exact_base)
        scope = _as_object(evidence.get("exception_scope"), "writer exception scope")
        return (
            evidence.get("kind") == "decision"
            and evidence.get("decision_type") == "exceptional_writer"
            and evidence.get("actor") == "maintainer"
            and evidence.get("approved") is True
            and isinstance(evidence.get("reference"), str)
            and bool(evidence["reference"])
            and _has_authoritative_decision(
                evidence,
                task["task_key"],
                identity,
                exact_base,
                task["maintainer_authorization"]["authority_reference"],
            )
            and scope.get("task_key") == task["task_key"]
            and _identity(scope.get("subject"), exact_base) == identity
            and scope.get("exact_base") == exact_base
            and set(
                normalized_path(path)
                for path in _as_list(scope.get("owned_paths"), "writer exception paths")
            )
            == set(task["owned_paths"])
            and scope.get("writer_mode") in {"root", "subagent"}
            and (writer_mode is None or scope.get("writer_mode") == writer_mode)
            and isinstance(scope.get("reason"), str)
            and bool(scope["reason"])
        )
    except (KeyError, WorkflowError):
        return False


def _require_writer_exception(
    state: dict[str, Any],
    task: dict[str, Any],
    record: dict[str, Any],
    writer_mode: str,
) -> dict[str, Any]:
    reference = record.get("writer_exception_reference")
    if not isinstance(reference, str) or not reference:
        raise WorkflowError(
            "exceptional writer activation requires a durable maintainer exception "
            "reference"
        )
    matches = [
        evidence
        for evidence in state["evidence"]
        if evidence.get("reference") == reference
        and _writer_exception_matches(
            evidence, task, state["repository"]["exact_base"], writer_mode
        )
    ]
    if len(matches) != 1:
        raise WorkflowError(
            "writer exception reference is not a task/subject-bound maintainer decision"
        )
    return matches[0]


def _structured_references(
    value: Any, label: str, task_key: str, identity: dict[str, str]
) -> list[dict[str, Any]]:
    references = _as_list(value, label)
    if not references:
        raise WorkflowError(f"{label} must not be empty")
    result = []
    for reference in references:
        reference = _as_object(reference, label)
        if reference.get("task_key") != task_key:
            raise WorkflowError(f"{label} must bind the handoff task")
        if (
            not isinstance(reference.get("reference"), str)
            or not reference["reference"]
        ):
            raise WorkflowError(f"{label} reference is required")
        if _identity(reference.get("identity"), identity["base"]) != identity:
            raise WorkflowError(f"{label} identity does not match the handoff")
        result.append(reference)
    return result


def validate_route(route: Any, task_class: str | None = None) -> None:
    route = _as_object(route, "route")
    declared_route = route.get("declared_route")
    declared_model = route.get("declared_model")
    declared_effort = route.get("declared_effort")
    selected_model = route.get("selected_model")
    selected_effort = route.get("selected_effort")
    if not isinstance(declared_route, str) or not declared_route:
        raise WorkflowError("declared route is required")
    if not isinstance(declared_model, str) or not declared_model:
        raise WorkflowError("declared model is required")
    if not isinstance(declared_effort, str) or not declared_effort:
        raise WorkflowError("declared reasoning effort is required")
    if not isinstance(selected_model, str) or not selected_model:
        raise WorkflowError(
            "selected model is required; prompt text is not route evidence"
        )
    if not isinstance(selected_effort, str) or not selected_effort:
        raise WorkflowError(
            "selected reasoning effort is required; prompt text is not route evidence"
        )
    if selected_model != declared_model or selected_effort != declared_effort:
        raise WorkflowError("selected route does not match the declared route")
    _validate_selection_mechanism(route.get("selection_mechanism"))
    observation = _as_object(route.get("runtime_observation"), "runtime_observation")
    availability = observation.get("availability")
    source = observation.get("source")
    if availability not in {"OBSERVED", "UNAVAILABLE"}:
        raise WorkflowError(
            "runtime observation availability must be OBSERVED or UNAVAILABLE"
        )
    if not isinstance(source, str) or not source:
        raise WorkflowError("runtime observation source is required")
    observed_model = observation.get("model")
    observed_effort = observation.get("effort")
    if availability == "OBSERVED":
        if not all(
            isinstance(value, str) and value and value != "UNVERIFIED"
            for value in (observed_model, observed_effort)
        ):
            raise WorkflowError(
                "observed runtime model and effort are required when telemetry "
                "is available"
            )
        expected_compliance = (
            "PASS"
            if observed_model == declared_model and observed_effort == declared_effort
            else "FAIL"
        )
    else:
        if observed_model != "UNVERIFIED" or observed_effort != "UNVERIFIED":
            raise WorkflowError(
                "unavailable runtime telemetry must use UNVERIFIED observed values"
            )
        expected_compliance = "UNVERIFIED"
    if route.get("declared_route_compliance") != expected_compliance:
        raise WorkflowError(
            "declared route compliance does not match runtime observation availability"
        )
    experiment = _as_object(route.get("experiment"), "route experiment")
    label = experiment.get("label")
    if label not in {
        "NOT_EXPERIMENT",
        "CONTROL",
        "EXPERIMENTAL",
        "OBSERVATIONAL",
    }:
        raise WorkflowError("unknown route experiment label")
    if label in {"CONTROL", "EXPERIMENTAL"}:
        if experiment.get("predeclared") is not True or not all(
            isinstance(experiment.get(name), str) and experiment[name]
            for name in (
                "trial_id",
                "pair_id",
                "run_id",
                "protocol_ref",
                "frozen_task_identity",
                "isolation_ref",
                "task_mix",
            )
        ):
            raise WorkflowError(
                "control and experimental runs require a predeclared matched pair "
                "with frozen-task and isolation metadata"
            )
        if experiment["protocol_ref"] != TRIAL_PROTOCOL_REF:
            raise WorkflowError(
                "matched run protocol_ref is not the Ultra trial protocol"
            )
        if experiment["task_mix"] not in TRIAL_TASK_MIX:
            raise WorkflowError("matched run task_mix is not part of the Ultra trial")
    elif experiment.get("predeclared") is not False:
        raise WorkflowError(
            "non-experimental and observational runs cannot claim predeclaration"
        )
    if task_class in PUBLIC_ROUTE:
        expected_route, efforts = PUBLIC_ROUTE[task_class]
        if label == "EXPERIMENTAL":
            route_allowed = declared_route == "sol-ultra" and declared_effort == "ultra"
        elif label == "CONTROL" and declared_route == "sol-high":
            route_allowed = declared_effort == "high"
        else:
            route_allowed = (
                declared_route == expected_route and declared_effort in efforts
            )
        if not route_allowed:
            raise WorkflowError(f"route is not permitted for task class {task_class}")
    marker = ROUTE_MODEL_MARKER.get(declared_route)
    if marker is None or marker not in declared_model.casefold():
        raise WorkflowError("declared model does not match the declared route family")


def _nonnegative_number(value: Any, label: str, *, integer: bool = False) -> float:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or value < 0:
        kind = "integer" if integer else "number"
        raise WorkflowError(f"{label} must be a non-negative {kind}")
    return float(value)


def _bounded_score(value: Any, label: str) -> float:
    score = _nonnegative_number(value, label)
    if score > 100:
        raise WorkflowError(f"{label} must be between 0 and 100")
    return score


def _close_score(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 0.01


def _clamp_score(value: float) -> float:
    return min(100.0, max(0.0, value))


def _validate_trial_score(score: Any, label: str) -> dict[str, Any]:
    score = _as_object(score, label)
    if not isinstance(score.get("ledger_ref"), str) or not score["ledger_ref"]:
        raise WorkflowError(f"{label} ledger_ref is required")
    raw = _as_object(score.get("raw"), f"{label} raw metrics")
    for name in ("input_tokens", "cached_input_tokens", "output_tokens"):
        _nonnegative_number(raw.get(name), f"{label} {name}", integer=True)
    _nonnegative_number(raw.get("elapsed_seconds"), f"{label} elapsed_seconds")
    for name in ("agent_sessions", "redundant_full_gate_runs"):
        _nonnegative_number(raw.get(name), f"{label} {name}", integer=True)
    for name in (
        "credited_ledger_weight",
        "total_ledger_weight",
        "false_positives",
        "avoidable_blocking_clarifications",
        "avoidable_tool_or_test_retries",
        "incomplete_outcome",
        "handoff_passed_items",
    ):
        _nonnegative_number(raw.get(name), f"{label} {name}", integer=True)
    if raw.get("acceptance_complete") not in {True, False}:
        raise WorkflowError(f"{label} acceptance_complete must be boolean")
    if raw["credited_ledger_weight"] > raw["total_ledger_weight"]:
        raise WorkflowError(f"{label} credited ledger weight exceeds the ledger")
    if raw["incomplete_outcome"] not in {0, 1}:
        raise WorkflowError(f"{label} incomplete_outcome must be zero or one")
    if raw["handoff_passed_items"] > 10:
        raise WorkflowError(f"{label} handoff_passed_items exceeds ten")
    expected_effective = (
        max(0, raw["input_tokens"] - raw["cached_input_tokens"]) + raw["output_tokens"]
    )
    if raw.get("effective_tokens") != expected_effective:
        raise WorkflowError(f"{label} effective_tokens does not match the formula")
    components = _as_object(score.get("components"), f"{label} score components")
    for name in ("quality", "autonomy", "efficiency", "handoff", "total"):
        _bounded_score(components.get(name), f"{label} {name}")
    evidence = _as_list(score.get("ledger_evidence_refs"), f"{label} ledger evidence")
    if not evidence or not all(isinstance(item, str) and item for item in evidence):
        raise WorkflowError(f"{label} ledger evidence must contain references")
    expected_components = {
        "autonomy": _clamp_score(
            100
            - 20 * raw["avoidable_blocking_clarifications"]
            - 10 * raw["avoidable_tool_or_test_retries"]
            - 25 * raw["incomplete_outcome"]
        ),
        "handoff": float(10 * raw["handoff_passed_items"]),
    }
    if raw["total_ledger_weight"] > 0:
        expected_components["quality"] = _clamp_score(
            100.0 * raw["credited_ledger_weight"] / raw["total_ledger_weight"]
            - min(20, 5 * raw["false_positives"])
        )
    for name, expected in expected_components.items():
        if not _close_score(float(components[name]), expected):
            raise WorkflowError(f"{label} {name} does not match raw scoring inputs")
    expected_total = (
        0.50 * components["quality"]
        + 0.20 * components["autonomy"]
        + 0.20 * components["efficiency"]
        + 0.10 * components["handoff"]
    )
    if not _close_score(float(components["total"]), expected_total):
        raise WorkflowError(f"{label} total does not match the normalized formula")
    return score


def _validate_pair_ledger(
    control: dict[str, Any], experimental: dict[str, Any], pair: dict[str, Any]
) -> None:
    pair_id = pair["pair_id"]
    ledger_ref = pair.get("ledger_ref")
    if not isinstance(ledger_ref, str) or not ledger_ref:
        raise WorkflowError(f"trial pair {pair_id} shared ledger_ref is required")
    runs = (control, experimental)
    if any(run["score"].get("ledger_ref") != ledger_ref for run in runs):
        raise WorkflowError(f"trial pair {pair_id} runs do not share one ledger")
    ledger_weights = {run["score"]["raw"]["total_ledger_weight"] for run in runs}
    if len(ledger_weights) != 1:
        raise WorkflowError(f"trial pair {pair_id} runs do not share one ledger weight")
    if next(iter(ledger_weights)) == 0:
        expected_quality = (
            100.0
            if all(run["score"]["raw"]["acceptance_complete"] for run in runs)
            else 0.0
        )
        if any(
            not _close_score(
                float(run["score"]["components"]["quality"]), expected_quality
            )
            for run in runs
        ):
            raise WorkflowError(
                f"trial pair {pair_id} zero-weight quality must use both runs' "
                "acceptance results"
            )


def _pair_metric_score(pair_minimum: float, run_value: float) -> float:
    if run_value == 0:
        return 100.0
    return 100.0 * pair_minimum / run_value


def _validate_pair_efficiency(
    control: dict[str, Any], experimental: dict[str, Any], pair_id: str
) -> None:
    weights = {
        "effective_tokens": 0.50,
        "elapsed_seconds": 0.30,
        "agent_sessions": 0.10,
        "redundant_full_gate_runs": 0.10,
    }
    runs = (control, experimental)
    for run in runs:
        raw = run["score"]["raw"]
        expected = 0.0
        for metric, weight in weights.items():
            pair_minimum = min(float(item["score"]["raw"][metric]) for item in runs)
            expected += weight * _pair_metric_score(pair_minimum, float(raw[metric]))
        actual = float(run["score"]["components"]["efficiency"])
        if not _close_score(actual, expected):
            raise WorkflowError(
                f"trial pair {pair_id} efficiency does not match paired raw metrics"
            )


def validate_route_trial(record: Any) -> dict[str, Any]:
    """Validate whether a completed route comparison may inform route defaults."""

    record = _as_object(record, "route trial")
    trial_id = record.get("trial_id")
    if not isinstance(trial_id, str) or not trial_id:
        raise WorkflowError("route trial_id is required")
    classification = record.get("classification")
    if classification == "OBSERVATIONAL":
        if record.get("eligible_for_default_route_decision") is not False:
            raise WorkflowError(
                "observational trials cannot support route-default decisions"
            )
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise WorkflowError("observational trial requires a reason")
        return copy.deepcopy(record)
    if classification != "MATCHED_TRIAL":
        raise WorkflowError("route trial classification is unknown")
    if record.get("status") != "COMPLETE":
        raise WorkflowError("matched route trial must be complete")
    if record.get("protocol_ref") != TRIAL_PROTOCOL_REF:
        raise WorkflowError("matched route trial must use the Ultra trial protocol")
    if record.get("eligible_for_default_route_decision") is not True:
        raise WorkflowError(
            "matched trial must explicitly claim route-default evidence eligibility"
        )
    adjudication = _as_object(record.get("adjudication"), "trial adjudication")
    if adjudication.get("blinded") is not True or not all(
        isinstance(adjudication.get(name), str) and adjudication[name]
        for name in ("alias_seed", "adjudicator_ref", "ledger_ref", "formula_ref")
    ):
        raise WorkflowError(
            "matched trial requires blinded, reproducible adjudication metadata"
        )
    if adjudication["formula_ref"] != TRIAL_PROTOCOL_REF:
        raise WorkflowError("trial scoring formula_ref is not the Ultra trial protocol")
    pairs = _as_list(record.get("pairs"), "route trial pairs")
    if len(pairs) != 6:
        raise WorkflowError("matched route trial requires exactly six pairs")
    pair_ids: set[str] = set()
    run_ids: set[str] = set()
    mix_counts = {name: 0 for name in TRIAL_TASK_MIX}
    total_deltas: list[float] = []
    quality_deltas: list[float] = []
    for pair_value in pairs:
        pair = _as_object(pair_value, "route trial pair")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise WorkflowError("route trial pair_id must be unique and non-empty")
        pair_ids.add(pair_id)
        task_mix = pair.get("task_mix")
        if task_mix not in mix_counts:
            raise WorkflowError("route trial pair has an unknown task_mix")
        mix_counts[task_mix] += 1
        if (
            not all(
                isinstance(pair.get(name), str) and pair[name]
                for name in (
                    "exact_base",
                    "frozen_task_identity",
                    "isolation_ref",
                )
            )
            or pair.get("isolation_verified") is not True
        ):
            raise WorkflowError(
                "route trial pair requires a frozen task and verified isolation"
            )
        runs = _as_list(pair.get("runs"), f"trial pair {pair_id} runs")
        if len(runs) != 2:
            raise WorkflowError("each route trial pair requires exactly two runs")
        by_label: dict[str, dict[str, Any]] = {}
        context_ids: set[str] = set()
        for run_value in runs:
            run = _as_object(run_value, f"trial pair {pair_id} run")
            run_id = run.get("run_id")
            context_id = run.get("context_identity")
            task_class = run.get("task_class")
            if (
                not isinstance(run_id, str)
                or not run_id
                or run_id in run_ids
                or not isinstance(context_id, str)
                or not context_id
                or context_id in context_ids
                or task_class not in PUBLIC_ROUTE
            ):
                raise WorkflowError(
                    "trial runs require unique run/context identities and a task class"
                )
            run_ids.add(run_id)
            context_ids.add(context_id)
            route = _as_object(run.get("route"), f"trial run {run_id} route")
            validate_route(route, task_class)
            experiment = route["experiment"]
            label = experiment["label"]
            if label not in {"CONTROL", "EXPERIMENTAL"} or label in by_label:
                raise WorkflowError(
                    "each route trial pair requires one control and one "
                    "experimental run"
                )
            if any(
                experiment.get(field) != expected
                for field, expected in {
                    "trial_id": trial_id,
                    "pair_id": pair_id,
                    "run_id": run_id,
                    "protocol_ref": record["protocol_ref"],
                    "frozen_task_identity": pair["frozen_task_identity"],
                    "isolation_ref": pair["isolation_ref"],
                    "task_mix": task_mix,
                }.items()
            ):
                raise WorkflowError(
                    "trial run route is not bound to its frozen pair metadata"
                )
            run["score"] = _validate_trial_score(
                run.get("score"), f"trial run {run_id} score"
            )
            by_label[label] = run
        if set(by_label) != {"CONTROL", "EXPERIMENTAL"}:
            raise WorkflowError(
                "each route trial pair requires one control and one experimental run"
            )
        if by_label["CONTROL"]["task_class"] != by_label["EXPERIMENTAL"]["task_class"]:
            raise WorkflowError("paired trial runs must use the same frozen task class")
        _validate_pair_ledger(by_label["CONTROL"], by_label["EXPERIMENTAL"], pair)
        _validate_pair_efficiency(
            by_label["CONTROL"], by_label["EXPERIMENTAL"], pair_id
        )
        total_deltas.append(
            float(by_label["EXPERIMENTAL"]["score"]["components"]["total"])
            - float(by_label["CONTROL"]["score"]["components"]["total"])
        )
        quality_deltas.append(
            float(by_label["EXPERIMENTAL"]["score"]["components"]["quality"])
            - float(by_label["CONTROL"]["score"]["components"]["quality"])
        )
    if mix_counts != TRIAL_TASK_MIX:
        raise WorkflowError("route trial does not match the required six-pair task mix")
    aggregate = _as_object(record.get("aggregate"), "route trial aggregate")
    for name in ("report_ref", "p1_p2_differences_ref"):
        if not isinstance(aggregate.get(name), str) or not aggregate[name]:
            raise WorkflowError(f"route trial aggregate {name} is required")
    expected_aggregate = {
        "mean_total_delta": statistics.fmean(total_deltas),
        "median_total_delta": statistics.median(total_deltas),
        "mean_quality_delta": statistics.fmean(quality_deltas),
    }
    for name, expected in expected_aggregate.items():
        actual = aggregate.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not _close_score(float(actual), expected)
        ):
            raise WorkflowError(f"route trial aggregate {name} is not reproducible")
    if not isinstance(aggregate.get("escaped_p1"), bool):
        raise WorkflowError("route trial aggregate escaped_p1 must be boolean")
    if aggregate.get("outcome") not in {
        "MATERIAL_QUALITY_BENEFIT",
        "NO_MATERIAL_BENEFIT",
        "TIE",
    }:
        raise WorkflowError("route trial aggregate outcome is unknown")
    if aggregate.get("recommendation") not in {
        "ADOPT_EXPERIMENTAL",
        "KEEP_CONTROL",
        "NO_CHANGE",
    }:
        raise WorkflowError("route trial aggregate recommendation is unknown")
    if aggregate.get("recommendation") == "ADOPT_EXPERIMENTAL" and (
        aggregate.get("outcome") != "MATERIAL_QUALITY_BENEFIT"
        or aggregate["escaped_p1"]
    ):
        raise WorkflowError(
            "experimental route adoption requires material quality benefit and no "
            "escaped P1"
        )
    return copy.deepcopy(record)


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
    owned_paths = [
        normalized_path(path) for path in _as_list(task["owned_paths"], "owned_paths")
    ]
    if not owned_paths or len(owned_paths) != len(set(owned_paths)):
        raise WorkflowError("task must own a non-empty, duplicate-free exact path list")
    if task.get("recursive_delegation") not in (None, False):
        raise WorkflowError("recursive delegation cannot be enabled")
    for name in (
        "acceptance_criteria",
        "non_goals",
        "verification",
        "review",
        "stop_conditions",
    ):
        if not _as_list(task[name], name):
            raise WorkflowError(f"task {name} must not be empty")
    validate_route(task["route"], task["task_class"])
    triggers = _as_list(task.get("review_triggers", []), "task review_triggers")
    if not all(isinstance(trigger, str) and trigger for trigger in triggers) or len(
        triggers
    ) != len(set(triggers)):
        raise WorkflowError("task review_triggers must be unique non-empty strings")
    authorization = _as_object(
        task.get("maintainer_authorization"), "task maintainer_authorization"
    )
    identity = _identity(task.get("subject"), task["exact_base"])
    if (
        authorization.get("actor") != "maintainer"
        or authorization.get("task_key") != key
        or authorization.get("exact_base") != task["exact_base"]
        or _identity(authorization.get("subject"), task["exact_base"]) != identity
        or _identity(authorization.get("task_packet_identity"), task["exact_base"])
        != identity
    ):
        raise WorkflowError(
            "task maintainer authorization is not bound to the task packet and subject"
        )
    if not all(
        isinstance(authorization.get(name), str) and authorization[name]
        for name in (
            "authority_reference",
            "readiness_evidence",
            "reviewed_corrections_evidence",
        )
    ):
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
    if (
        not isinstance(repository.get("exact_base"), str)
        or not repository["exact_base"]
    ):
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
        paths = [
            normalized_path(path)
            for path in _as_list(lease.get("paths"), "ownership paths")
        ]
        overlap = [
            (existing, path)
            for existing in active_paths
            for path in paths
            if paths_overlap(existing, path)
        ]
        if overlap:
            raise WorkflowError(f"active ownership path collision: {overlap!r}")
        active_paths.extend(paths)
    evidence_records = [
        _as_object(item, "evidence")
        for item in _as_list(state.get("evidence"), "evidence")
    ]
    workers = _as_list(state.get("workers"), "workers")
    seen_worker_refs: set[str] = set()
    active_worker_tasks: set[str] = set()
    active_writers: list[dict[str, Any]] = []
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
        if worker.get("status") == "SPAWN_UNKNOWN" and (
            not isinstance(worker.get("ambiguity_reference"), str)
            or not worker["ambiguity_reference"]
        ):
            raise WorkflowError(
                "ambiguous worker state requires durable ambiguity_reference"
            )
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
                raise WorkflowError(
                    "worker write capability does not match its task class"
                )
            if derived_writer and key not in active_keys:
                raise WorkflowError("active writer has no active ownership lease")
            if derived_writer:
                active_writers.append(worker)
            if key in active_worker_tasks:
                raise WorkflowError(
                    "more than one active or spawn-requested worker exists for a task"
                )
            active_worker_tasks.add(key)
        execution_mode = worker.get("execution_mode", "subagent")
        if execution_mode not in {"root", "subagent"}:
            raise WorkflowError("worker execution_mode must be root or subagent")
        if (
            execution_mode == "root"
            and worker.get("reference") != f"root:{worker.get('task_key')}"
        ):
            raise WorkflowError("root writer reference must be task-specific")
        if execution_mode == "subagent" and worker.get("writer") is True:
            task = tasks.get(worker.get("task_key"))
            if (
                task is None
                or len(
                    [
                        evidence
                        for evidence in evidence_records
                        if evidence.get("reference")
                        == worker.get("writer_exception_reference")
                        and _writer_exception_matches(
                            evidence,
                            task,
                            repository["exact_base"],
                            execution_mode,
                        )
                    ]
                )
                != 1
            ):
                raise WorkflowError(
                    "writer subagent lacks a durable task/subject-bound maintainer "
                    "exception"
                )
        _validate_monitor(worker.get("monitor"), worker["status"])
    if _active_writer_count(state) > 2:
        raise WorkflowError("a third active implementation writer is prohibited")
    if len(active_writers) == 2:
        guards = active_writers[1].get("independence_guards")
        if not isinstance(guards, dict) or not all(
            guards.get(name) is True for name in REQUIRED_SECOND_WRITER_GUARDS
        ):
            raise WorkflowError(
                "second active implementation writer requires every recorded "
                "independence guard"
            )
        second = active_writers[1]
        second_task = tasks[second["task_key"]]
        if (
            len(
                [
                    evidence
                    for evidence in evidence_records
                    if evidence.get("reference")
                    == second.get("writer_exception_reference")
                    and _writer_exception_matches(
                        evidence,
                        second_task,
                        repository["exact_base"],
                        second.get("execution_mode", "subagent"),
                    )
                ]
            )
            != 1
        ):
            raise WorkflowError(
                "second active implementation writer lacks a durable maintainer "
                "exception"
            )
    for field in (
        "evidence",
        "verification",
        "checkpoints",
        "handoffs",
        "reviews",
    ):
        _as_list(state.get(field), field)
    route_trials = _as_list(state.get("route_trials", []), "route_trials")
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
                raise WorkflowError(
                    "decision evidence lacks authoritative task/subject provenance"
                )
        elif evidence.get("kind") == "verifier_registration":
            raise WorkflowError("final-verifier agent registrations are prohibited")
    final_gates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for verification in state["verification"]:
        verification = _as_object(verification, "verification record")
        key = verification.get("task_key")
        if key not in tasks:
            raise WorkflowError("verification must bind a registered task_key")
        _verification_request(verification)
        _identity(verification.get("subject"), repository["exact_base"])
        if verification.get("status") not in {
            "PASS",
            "FAIL",
            "UNKNOWN",
            "NON_REUSABLE",
        }:
            raise WorkflowError("unknown verification status")
        if "verifier_reference" in verification:
            raise WorkflowError("final-verifier agent references are prohibited")
        subject = _identity(verification.get("subject"), repository["exact_base"])
        if (
            verification.get("command") == FINAL_GATE_COMMAND
            or verification.get("verification_type") == "public_safety"
        ) and verification.get("executor") != ROOT_EXECUTOR:
            raise WorkflowError(
                "final gate and public-safety checks must be root-executed"
            )
        if verification.get("command") == FINAL_GATE_COMMAND:
            subject_value = subject.get("head", subject.get("patch", ""))
            gate_key = (key, subject_value)
            prior_gates = final_gates.get(gate_key, [])
            if prior_gates and not (
                verification.get("rerun_reason") == "EVIDENCE_LOSS"
                and all(item.get("invalidated") is True for item in prior_gates)
            ):
                raise WorkflowError(
                    "duplicate full gate exists for one immutable candidate revision"
                )
            final_gates.setdefault(gate_key, []).append(verification)
    review_keys: set[str] = set()
    for review in state["reviews"]:
        review = _as_object(review, "review")
        key = review.get("task_key")
        if key not in tasks or key in review_keys:
            raise WorkflowError("review must bind one registered task")
        review_keys.add(key)
        if review.get("status") not in {
            "REQUIRED",
            "REPAIR_IN_PROGRESS",
            "PASS",
        }:
            raise WorkflowError("unknown review status")
        repair_cycles = review.get("repair_cycles")
        if (
            review.get("reviewers") != 1
            or isinstance(repair_cycles, bool)
            or not isinstance(repair_cycles, int)
            or repair_cycles < 0
        ):
            raise WorkflowError(
                "review must retain one reviewer and a non-negative repair count"
            )
        if not isinstance(review.get("subject"), dict):
            raise WorkflowError("review must bind an exact subject")
        if (
            review.get("route") != "terra-high"
            or review.get("write_ownership") != "none"
        ):
            raise WorkflowError("review must use the cold Terra-high read-only route")
        if not isinstance(review.get("rubric_ref"), str) or not review["rubric_ref"]:
            raise WorkflowError("review rubric reference is required")
    trial_ids: set[str] = set()
    for trial in route_trials:
        validated_trial = validate_route_trial(trial)
        if validated_trial["trial_id"] in trial_ids:
            raise WorkflowError("route trial_id is already registered")
        trial_ids.add(validated_trial["trial_id"])
    return state


def transition(state: dict[str, Any], target: str) -> dict[str, Any]:
    validate_state(state)
    if target not in ALLOWED_TRANSITIONS[state["phase"]]:
        raise WorkflowError(f"invalid transition {state['phase']} -> {target}")
    if target == "INTEGRATE":
        raise WorkflowError(
            "integration transition requires maintainer integration authorization"
        )
    if target not in AUTHORIZATION_PHASES[state["authorization_state"]]:
        raise WorkflowError("authorization state is incompatible with workflow phase")
    state["phase"] = target
    return {"phase": target}


def _transition_with_authorization(
    state: dict[str, Any], target: str, authorization_state: str
) -> None:
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
        if state["phase"] != "RECONCILE" or state["authorization_state"] not in {
            "PROPOSED",
            "REVIEWED_WITH_CORRECTIONS",
            "MAINTAINER_AUTHORIZED",
        }:
            raise WorkflowError(
                "maintainer authorization is only valid from reconciliation"
            )
    else:
        _authorize_integration(state, record)
    state["authorization_state"] = target
    if target == "INTEGRATION_AUTHORIZED":
        state["phase"] = "INTEGRATE"
    return {
        "authorization_state": target,
        "actor": "maintainer",
        "phase": state["phase"],
    }


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
        raise WorkflowError(
            "authoritative state is not currently maintainer-authorized"
        )


def acquire_ownership(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if state["authorization_state"] not in {
        "MAINTAINER_AUTHORIZED",
        "IMPLEMENTATION_READY",
    }:
        raise WorkflowError(
            "authoritative state is not currently ready for implementation ownership"
        )
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
        if (
            lease.get("active")
            and lease.get("task_key") == key
            and set(lease.get("paths", [])) == set(paths)
        ):
            state["authorization_state"] = "IMPLEMENTATION_READY"
            return lease
    for lease in state["ownership"]:
        if lease.get("active") and (
            any(
                paths_overlap(normalized_path(existing), path)
                for existing in lease["paths"]
                for path in paths
            )
        ):
            raise WorkflowError("task-key or path ownership collision")
    lease = {"task_key": key, "paths": paths, "active": True}
    state["ownership"].append(lease)
    state["authorization_state"] = "IMPLEMENTATION_READY"
    return lease


def release_ownership(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    key = record.get("task_key")
    if any(
        worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES
        for worker in state["workers"]
    ):
        raise WorkflowError(
            "ownership cannot be released while a task worker is active"
        )
    for lease in state["ownership"]:
        if lease.get("task_key") == key and lease.get("active"):
            lease["active"] = False
            return {"task_key": key, "released": True}
    raise WorkflowError("no active ownership lease for task")


def _active_writer_count(state: dict[str, Any]) -> int:
    return sum(
        worker.get("writer") and worker.get("status") in ACTIVE_WORKER_STATES
        for worker in state["workers"]
    )


def _check_writer_capacity(
    state: dict[str, Any],
    task: dict[str, Any],
    record: dict[str, Any],
    writer_mode: str,
) -> dict[str, Any] | None:
    count = _active_writer_count(state)
    if count >= 2:
        raise WorkflowError("a third active implementation writer is prohibited")
    if count == 1:
        guards = record.get("independence_guards")
        if not isinstance(guards, dict) or not all(
            guards.get(name) is True for name in REQUIRED_SECOND_WRITER_GUARDS
        ):
            raise WorkflowError(
                "second writer requires every recorded independence guard"
            )
        return _require_writer_exception(state, task, record, writer_mode)
    return None


def _activate_root_writer(
    state: dict[str, Any], task: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Activate the root task as the default implementation writer."""

    if task["route"]["declared_route_compliance"] == "FAIL":
        raise WorkflowError("observed route mismatch blocks implementation")
    key = task["task_key"]
    reference = f"root:{key}"
    existing = next(
        (worker for worker in state["workers"] if worker.get("reference") == reference),
        None,
    )
    if existing is not None:
        if existing.get("status") not in {"COMPLETE", "RETIRED"}:
            raise WorkflowError("task already has an active root writer")
        exception = _check_writer_capacity(state, task, record, "root")
        existing.update(
            {
                "task_class": task["task_class"],
                "writer": True,
                "status": "ACTIVE",
                "route": copy.deepcopy(task["route"]),
                "attempt": existing.get("attempt", 1) + 1,
                "execution_mode": "root",
                "writer_exception_reference": (
                    exception["reference"] if exception is not None else None
                ),
                "independence_guards": copy.deepcopy(record.get("independence_guards")),
            }
        )
        _reset_monitor(existing["monitor"], "ACTIVE")
        return existing
    exception = _check_writer_capacity(state, task, record, "root")
    root = {
        "reference": reference,
        "task_key": key,
        "task_class": task["task_class"],
        "writer": True,
        "status": "ACTIVE",
        "route": copy.deepcopy(task["route"]),
        "attempt": 1,
        "execution_mode": "root",
        "writer_exception_reference": (
            exception["reference"] if exception is not None else None
        ),
        "independence_guards": copy.deepcopy(record.get("independence_guards")),
        "monitor": _new_monitor("ACTIVE"),
    }
    state["workers"].append(root)
    return root


def register_worker(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if state["authorization_state"] != "IMPLEMENTING" or state["phase"] != "DISPATCH":
        raise WorkflowError(
            "worker registration is only valid for an implementing dispatch"
        )
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
    if writer and not any(
        lease.get("active") and lease.get("task_key") == key
        for lease in state["ownership"]
    ):
        raise WorkflowError("worker requires an active ownership lease")
    reference = record.get("reference")
    if not isinstance(reference, str) or not reference:
        raise WorkflowError("worker reference is required")
    if any(worker["reference"] == reference for worker in state["workers"]):
        raise WorkflowError("worker reference is already registered")
    if any(
        worker.get("task_key") == key and worker.get("status") == "SPAWN_UNKNOWN"
        for worker in state["workers"]
    ):
        raise WorkflowError(
            "SPAWN_UNKNOWN blocks retry until a maintainer replacement decision"
        )
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
        raise WorkflowError(
            "worker registration requires a durable task-specific "
            "SPAWN_REQUESTED intent"
        )
    if any(
        worker is not manual_intent
        and worker.get("task_key") == key
        and worker.get("status") in ACTIVE_WORKER_STATES
        for worker in state["workers"]
    ):
        raise WorkflowError("task already has an active worker")
    manual_intent["reference"] = reference
    manual_intent["status"] = "ACTIVE"
    manual_intent["execution_mode"] = "subagent"
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
            if (
                worker.get("status") == "SPAWN_UNKNOWN"
                and record.get("maintainer_replacement_decision") is not True
            ):
                raise WorkflowError(
                    "SPAWN_UNKNOWN requires a maintainer replacement decision"
                )
            if target == "SPAWN_UNKNOWN":
                if (
                    not isinstance(record.get("ambiguity_reference"), str)
                    or not record["ambiguity_reference"]
                ):
                    raise WorkflowError(
                        "SPAWN_UNKNOWN requires durable ambiguity_reference"
                    )
                worker["ambiguity_reference"] = record["ambiguity_reference"]
            worker["status"] = target
            _reset_monitor(worker["monitor"], target)
            return {"reference": reference, "status": target}
        if worker.get("status") == "SPAWN_UNKNOWN":
            if record.get("maintainer_replacement_decision") is not True:
                return {
                    "reference": reference,
                    "status": "SPAWN_UNKNOWN",
                    "retry": "BLOCKED",
                }
            worker["status"] = "RETIRED"
            _reset_monitor(worker["monitor"], "RETIRED")
            return {
                "reference": reference,
                "status": "RETIRED",
                "replacement": "MAINTAINER_APPROVED",
            }
        return {
            "reference": reference,
            "status": worker["status"],
            "retry": "NOT_NEEDED",
        }
    raise WorkflowError("unknown worker reference")


def monitor_worker(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Persist deterministic, low-frequency polling guidance for one worker.

    This is deliberately a record operation, not a timer or background service.
    Callers provide an integer clock value and may wait longer than the returned
    interval without changing the result's safety properties.
    """

    validate_state(state)
    reference = record.get("reference")
    worker = next(
        (item for item in state["workers"] if item.get("reference") == reference), None
    )
    if worker is None:
        raise WorkflowError("unknown worker reference")
    now = record.get("now")
    if not isinstance(now, int) or now < 0:
        raise WorkflowError("worker monitor requires a non-negative integer now")
    monitor = _validate_monitor(worker["monitor"], worker["status"])
    if record.get("system_error") is True:
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now
        return {
            "action": "WAKE_IMMEDIATELY",
            "reason": "SYSTEM_ERROR",
            "reference": reference,
        }
    if record.get("maintainer_input") is True:
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now
        return {
            "action": "WAKE_IMMEDIATELY",
            "reason": "MAINTAINER_INPUT",
            "reference": reference,
        }
    observed_status = record.get("status", worker["status"])
    if observed_status not in ACTIVE_WORKER_STATES | {"COMPLETE", "RETIRED"}:
        raise WorkflowError("unknown monitored worker status")
    if worker["status"] == "SPAWN_UNKNOWN":
        return {
            "action": "WAKE_IMMEDIATELY",
            "reason": "AMBIGUOUS_SPAWN_RECONCILIATION",
            "reference": reference,
        }
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
                raise WorkflowError(
                    "SPAWN_UNKNOWN requires durable ambiguity_reference"
                )
            worker["ambiguity_reference"] = ambiguity
        worker["status"] = observed_status
        _reset_monitor(monitor, observed_status)
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = (
            now
            if observed_status in {"COMPLETE", "SPAWN_UNKNOWN"}
            else now + HEARTBEAT_DEFAULT_SECONDS
        )
        if observed_status in {"COMPLETE", "SPAWN_UNKNOWN", "RETIRED"}:
            reason = (
                "COMPLETION"
                if observed_status == "COMPLETE"
                else "AMBIGUOUS_SPAWN_RECONCILIATION"
                if observed_status == "SPAWN_UNKNOWN"
                else "TERMINAL_STATUS"
            )
            return {
                "action": "WAKE_IMMEDIATELY",
                "reason": reason,
                "reference": reference,
            }
        return {
            "action": "HEARTBEAT_CHANGED",
            "reference": reference,
            "interval_seconds": monitor["interval_seconds"],
            "next_check_at": monitor["next_check_at"],
        }
    if worker["status"] in {"COMPLETE", "RETIRED"}:
        return {
            "action": "WAKE_IMMEDIATELY",
            "reason": "TERMINAL_STATUS",
            "reference": reference,
        }
    for name in ("output_identity", "repository_evidence"):
        if (
            name in record
            and record[name] is not None
            and (not isinstance(record[name], str) or not record[name])
        ):
            raise WorkflowError(
                f"worker monitor {name} must be a non-empty string when supplied"
            )
    output_identity = record.get("output_identity", monitor["last_output_identity"])
    repository_evidence = record.get(
        "repository_evidence", monitor["last_repository_evidence"]
    )
    changed = (
        output_identity != monitor["last_output_identity"]
        or repository_evidence != monitor["last_repository_evidence"]
    )
    if changed:
        _reset_monitor(monitor, worker["status"])
        monitor["last_output_identity"] = output_identity
        monitor["last_repository_evidence"] = repository_evidence
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now + HEARTBEAT_DEFAULT_SECONDS
        return {
            "action": "HEARTBEAT_CHANGED",
            "reference": reference,
            "interval_seconds": monitor["interval_seconds"],
            "next_check_at": monitor["next_check_at"],
        }
    if monitor["last_heartbeat_at"] is None:
        monitor["last_heartbeat_at"] = now
        monitor["next_check_at"] = now + monitor["interval_seconds"]
    elif now >= monitor["next_check_at"]:
        monitor["interval_seconds"] = min(
            monitor["interval_seconds"] * 2, monitor["max_interval_seconds"]
        )
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
        provenance = _as_object(
            record.get("authoritative_provenance"), "decision authoritative_provenance"
        )
        accepted_authority = state["tasks"][key]["maintainer_authorization"][
            "authority_reference"
        ]
        if (
            provenance.get("task_key") != key
            or _identity(provenance.get("subject"), state["repository"]["exact_base"])
            != subject
            or provenance.get("authority_reference") != accepted_authority
        ):
            raise WorkflowError(
                "decision evidence lacks authoritative task/subject provenance"
            )
        if record.get("decision_type") == "exceptional_writer":
            if not _writer_exception_matches(
                record, state["tasks"][key], state["repository"]["exact_base"]
            ):
                raise WorkflowError(
                    "writer exception must be a task/subject-bound maintainer decision"
                )
            if any(
                evidence.get("reference") == record["reference"]
                for evidence in state["evidence"]
            ):
                raise WorkflowError("writer exception reference is already registered")
    elif record["kind"] == "verifier_registration":
        raise WorkflowError("final-verifier agent registrations are prohibited")
    stored = copy.deepcopy(record)
    stored["sequence"] = _next_sequence(state)
    state["evidence"].append(stored)
    return {"recorded": "evidence", "count": len(state["evidence"])}


def record_route_trial(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    trial = validate_route_trial(record)
    trials = state.setdefault("route_trials", [])
    if any(existing.get("trial_id") == trial["trial_id"] for existing in trials):
        raise WorkflowError("route trial_id is already registered")
    stored = copy.deepcopy(trial)
    stored["sequence"] = _next_sequence(state)
    trials.append(stored)
    return {
        "recorded": "route_trial",
        "trial_id": trial["trial_id"],
        "eligible_for_default_route_decision": trial[
            "eligible_for_default_route_decision"
        ],
    }


def record_verification(
    state: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    validate_state(state)
    if state["phase"] != "VERIFY" or state["authorization_state"] not in {
        "IMPLEMENTING",
        "VERIFIED",
    }:
        raise WorkflowError("verification recording is only valid in VERIFY")
    key = record.get("task_key")
    if key not in state["tasks"]:
        raise WorkflowError("verification requires a registered task_key")
    for name in (
        "reference",
        "subject",
        "command",
        "scope",
        "environment",
        "side_effects",
        "status",
    ):
        if name not in record:
            raise WorkflowError(f"verification {name} is required")
    if not isinstance(record["reference"], str) or not record["reference"]:
        raise WorkflowError("verification reference is required")
    if record["status"] not in {"PASS", "FAIL", "UNKNOWN", "NON_REUSABLE"}:
        raise WorkflowError("unknown verification status")
    subject = _identity(record["subject"], state["repository"]["exact_base"])
    request = _verification_request(record)
    if "verifier_reference" in record:
        raise WorkflowError("final-verifier agent references are prohibited")
    is_final_gate = record["command"] == FINAL_GATE_COMMAND
    is_public_safety = record.get("verification_type") == "public_safety"
    if (is_final_gate or is_public_safety) and record.get("executor") != ROOT_EXECUTOR:
        raise WorkflowError("final gate and public-safety checks must be root-executed")
    if is_final_gate:
        handoff = _latest_handoff(state, key)
        if handoff is None or handoff.get("identity") != subject:
            raise WorkflowError("final gate requires the exact final assembled handoff")
        review = _review_for_task(state, key)
        if review_required(state["tasks"][key])["required"] and (
            review is None
            or review.get("status") != "PASS"
            or review.get("subject") != subject
        ):
            raise WorkflowError(
                "final gate requires a passing exact-subject cold review"
            )
        prior_gates = [
            existing
            for existing in state["verification"]
            if existing.get("task_key") == key
            and existing.get("subject") == subject
            and existing.get("command") == FINAL_GATE_COMMAND
        ]
        if prior_gates and not (
            record.get("rerun_reason") == "EVIDENCE_LOSS"
            and all(existing.get("invalidated") is True for existing in prior_gates)
        ):
            raise WorkflowError(
                "full gate already recorded for this immutable candidate revision"
            )
    for existing in state["verification"]:
        if existing.get("reference") == record["reference"] and not (
            existing.get("task_key") == key
            and _identity(existing.get("subject"), state["repository"]["exact_base"])
            == subject
            and _verification_request(existing) == request
        ):
            raise WorkflowError(
                "verification reference must be unique to task, subject, and "
                "request identity"
            )
    stored = copy.deepcopy(record)
    stored["subject"] = subject
    stored["sequence"] = _next_sequence(state)
    state["verification"].append(stored)
    _refresh_verification_authorization(state)
    return {"recorded": "verification", "count": len(state["verification"])}


def verification_reuse(
    record: dict[str, Any], subject: dict[str, Any], requested: Any = None
) -> dict[str, Any]:
    record = _as_object(record, "verification record")
    subject = _as_object(subject, "subject")
    required = (
        "task_key",
        "subject",
        "command",
        "scope",
        "environment",
        "side_effects",
        "status",
    )
    if any(name not in record for name in required) or record.get("status") != "PASS":
        return {"reusable": False, "status": "NON_REUSABLE"}
    try:
        exact_base = subject.get("base")
        if not isinstance(exact_base, str) or _identity(
            subject, exact_base
        ) != _identity(record["subject"], exact_base):
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
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) and item[key]
            for key in ("content_identity", "role", "scope")
        ):
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
        "mission": {
            "priority": state["mission"]["priority"],
            "mission_fit": state["mission"]["mission_fit"],
        },
        "phase": state["phase"],
        "repository": {"exact_base": state["repository"]["exact_base"]},
        "tasks": [
            {
                "task_key": task["task_key"],
                "objective": task["objective"],
                "owned_paths": task["owned_paths"],
                "stop_conditions": task["stop_conditions"],
            }
            for task in state["tasks"].values()
        ],
        "active_ownership": [
            lease for lease in state["ownership"] if lease.get("active")
        ],
        "active_workers": [
            {
                "reference": worker["reference"],
                "task_key": worker["task_key"],
                "status": worker["status"],
            }
            for worker in state["workers"]
            if worker["status"] in ACTIVE_WORKER_STATES
        ],
        "verification": [
            {"status": item.get("status"), "subject": item.get("subject")}
            for item in state["verification"]
        ],
        "next_action": next_action(state),
    }


def ingest_handoff(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    if state["phase"] != "DISPATCH" or state["authorization_state"] != "IMPLEMENTING":
        raise WorkflowError(
            "handoff ingestion is only valid in an implementing dispatch"
        )
    key = record.get("task_key")
    task = state["tasks"].get(key)
    if task is None:
        raise WorkflowError("handoff task is not registered")
    worker_reference = record.get("worker_reference")
    if not isinstance(worker_reference, str) or not worker_reference:
        raise WorkflowError("handoff worker_reference is required")
    worker = next(
        (
            worker
            for worker in state["workers"]
            if worker.get("reference") == worker_reference
            and worker.get("task_key") == key
        ),
        None,
    )
    if worker is None:
        raise WorkflowError("handoff worker reference does not belong to the task")
    if worker.get("status") != "ACTIVE":
        raise WorkflowError("handoff worker is not active")
    changed = [
        normalized_path(path)
        for path in _as_list(record.get("changed_paths"), "changed_paths")
    ]
    if not set(changed).issubset(task["owned_paths"]):
        raise WorkflowError("handoff changed paths exceed task ownership")
    for name in (
        "identity",
        "route",
        "acceptance_evidence",
        "verification",
        "applied_decisions",
        "verification_references",
        "scope_safety",
        "open_findings",
        "integration_guidance",
    ):
        if name not in record:
            raise WorkflowError(f"handoff {name} is required")
    identity = _identity(record["identity"], state["repository"]["exact_base"])
    applied_decisions = _structured_references(
        record["applied_decisions"], "handoff applied_decisions", key, identity
    )
    verification_references = _structured_references(
        record["verification_references"],
        "handoff verification_references",
        key,
        identity,
    )
    for reference in verification_references:
        if "verification_request" not in reference:
            raise WorkflowError(
                "handoff verification_references must bind a request identity"
            )
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
            raise WorkflowError(
                "handoff applied decision is not an authoritative task decision"
            )
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


def review_required(
    task: dict[str, Any], triggers: list[str] | None = None
) -> dict[str, Any]:
    task = validate_task(task)
    if task["task_class"] == "protected_boundary_implementation":
        return {
            "required": True,
            "reviewers": 1,
            "route": "terra-high",
            "reason": "protected_boundary",
        }
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
        "PLAN": "ACQUIRE_OWNERSHIP_AND_START_ROOT_IMPLEMENTATION",
        "DISPATCH": "ROOT_IMPLEMENTATION_OR_EXCEPTIONAL_WORKER_EVENT",
        "VERIFY": "REUSE_OR_RUN_REQUIRED_VERIFICATION",
        "INTEGRATE": "AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION",
        "CLOSE": "STOP",
    }
    return actions[state["phase"]]


def _advance_result(
    state: dict[str, Any],
    *,
    advanced: bool,
    next_step: str,
    transitions: list[str],
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "advanced": advanced,
        "phase": state["phase"],
        "authorization_state": state["authorization_state"],
        "next_action": next_step,
        "automatic_transitions": transitions,
    }
    result.update(extra)
    return result


def _latest_handoff(state: dict[str, Any], task_key: str) -> dict[str, Any] | None:
    return next(
        (
            handoff
            for handoff in reversed(state["handoffs"])
            if handoff.get("task_key") == task_key
        ),
        None,
    )


def _review_for_task(state: dict[str, Any], task_key: str) -> dict[str, Any] | None:
    return next(
        (review for review in state["reviews"] if review.get("task_key") == task_key),
        None,
    )


def _reusable_verification(
    state: dict[str, Any], task_key: str, subject: dict[str, Any], requested: Any
) -> dict[str, Any] | None:
    verification = _latest_verification(state, task_key, subject, requested)
    if verification is None:
        return None
    return (
        verification
        if verification_reuse(verification, subject, requested)["reusable"]
        else None
    )


def _latest_verification(
    state: dict[str, Any], task_key: str, subject: dict[str, Any], requested: Any
) -> dict[str, Any] | None:
    try:
        request_identity = _verification_request(
            _as_object(requested, "verification request")
        )
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
    if state["phase"] != "VERIFY" or state["authorization_state"] not in {
        "IMPLEMENTING",
        "VERIFIED",
    }:
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
            if (
                latest is not None
                and verification_reuse(latest, subject, requested)["reusable"]
                and _matching_handoff_verification(handoff, latest)
            ):
                state["authorization_state"] = "VERIFIED"
                return
    state["authorization_state"] = "IMPLEMENTING"


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


def _matching_handoff_verification(
    handoff: dict[str, Any], verification: dict[str, Any]
) -> bool:
    reference = verification.get("reference")
    return any(
        item.get("reference") == reference
        and item.get("task_key") == handoff["task_key"]
        and item.get("identity") == handoff["identity"]
        and verification.get("task_key") == handoff["task_key"]
        and verification.get("subject") == handoff["identity"]
        and _verification_request(item.get("verification_request", {}))
        == _verification_request(verification)
        for item in handoff.get("verification_references", [])
    )


def _authorize_integration(state: dict[str, Any], record: dict[str, Any]) -> None:
    if state["phase"] != "VERIFY" or state["authorization_state"] != "VERIFIED":
        raise WorkflowError(
            "integration authorization requires a verified candidate in VERIFY"
        )
    key = record.get("task_key")
    if key not in state["tasks"]:
        raise WorkflowError("integration authorization requires a registered task_key")
    handoff = _latest_handoff(state, key)
    if handoff is None:
        raise WorkflowError("integration authorization requires a completed handoff")
    identity = _identity(record.get("identity"), state["repository"]["exact_base"])
    if identity != handoff["identity"]:
        raise WorkflowError(
            "integration authorization identity does not match the handoff"
        )
    decision = _as_object(record.get("applied_decision"), "applied_decision")
    if decision not in handoff["applied_decisions"]:
        raise WorkflowError(
            "integration authorization requires a matching applied decision"
        )
    reference = record.get("verification_reference")
    if not isinstance(reference, str) or not reference:
        raise WorkflowError(
            "integration authorization requires a verification reference"
        )
    requested = record.get("verification_request")
    verification = _reusable_verification(state, key, identity, requested)
    if verification is not None and verification.get("reference") != reference:
        verification = None
    if verification is None or not _matching_handoff_verification(
        handoff, verification
    ):
        raise WorkflowError(
            "integration authorization requires a handoff-bound verification reference"
        )
    if verification_reuse(verification, identity, requested)["reusable"] is not True:
        raise WorkflowError(
            "integration authorization requires a matching reusable PASS verification"
        )
    review = _review_for_task(state, key)
    if review_required(state["tasks"][key])["required"] and review is None:
        raise WorkflowError(
            "integration authorization requires a passing review of the exact subject"
        )
    if review is not None and (
        review.get("status") != "PASS" or review.get("subject") != identity
    ):
        raise WorkflowError(
            "integration authorization requires a passing review of the exact subject"
        )
    review_sequence = (
        review.get("completed_sequence", review.get("sequence", 0))
        if review is not None
        else 0
    )
    final_gate = None
    for item in reversed(state["verification"]):
        if (
            item.get("task_key") != key
            or item.get("subject") != identity
            or item.get("command") != ["python", "scripts/temper-gate.py", "all"]
        ):
            continue
        final_request = {
            name: item[name]
            for name in ("command", "scope", "environment", "side_effects")
        }
        if (
            _latest_verification(state, key, identity, final_request) is item
            and item.get("status") == "PASS"
            and item.get("sequence", 0) > handoff.get("sequence", 0)
            and item.get("sequence", 0) > review_sequence
            and item.get("executor") == ROOT_EXECUTOR
        ):
            final_gate = item
            break
    if final_gate is None:
        raise WorkflowError(
            "integration authorization requires current exact-subject "
            "root-executed temper-gate all PASS evidence after final assembly"
        )
    public_safety = None
    for item in reversed(state["verification"]):
        if (
            item.get("task_key") != key
            or item.get("subject") != identity
            or item.get("verification_type") != "public_safety"
        ):
            continue
        public_request = {
            name: item[name]
            for name in ("command", "scope", "environment", "side_effects")
        }
        if (
            _latest_verification(state, key, identity, public_request) is item
            and item.get("status") == "PASS"
            and item.get("executor") == ROOT_EXECUTOR
        ):
            public_safety = item
            break
    if public_safety is None:
        raise WorkflowError(
            "integration authorization requires exact-subject public-safety "
            "PASS evidence"
        )


def _create_dispatch_intent(
    state: dict[str, Any], task: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    key = task["task_key"]
    if any(
        worker.get("task_key") == key and worker.get("status") in ACTIVE_WORKER_STATES
        for worker in state["workers"]
    ):
        raise WorkflowError("task already has an active or spawn-requested worker")
    writer = TASK_WRITE_CAPABILITY[task["task_class"]]
    if not writer:
        raise WorkflowError("read-only task class cannot dispatch a writer")
    exception = _require_writer_exception(state, task, record, "subagent")
    if task["route"]["declared_route_compliance"] == "FAIL":
        raise WorkflowError("observed route mismatch blocks implementation")
    if "writer" in record and record["writer"] is not writer:
        raise WorkflowError("worker write capability is derived from task class")
    supplied_route = record.get("route")
    if supplied_route is not None:
        validate_route(supplied_route, task["task_class"])
        if not _routes_match(supplied_route, task["route"]):
            raise WorkflowError("dispatch supplied route does not match its task")
    if not any(
        lease.get("active") and lease.get("task_key") == key
        for lease in state["ownership"]
    ):
        raise WorkflowError("dispatch requires an active ownership lease")
    validate_route(task["route"], task["task_class"])
    _check_writer_capacity(state, task, record, "subagent")
    intent = {
        "reference": f"manual:{key}",
        "task_key": key,
        "task_class": task["task_class"],
        "writer": writer,
        "status": "SPAWN_REQUESTED",
        "route": copy.deepcopy(task["route"]),
        "attempt": record.get("attempt", 1),
        "execution_mode": "subagent",
        "writer_exception_reference": exception["reference"],
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
        return _advance_result(
            state,
            advanced=False,
            next_step="AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION",
            transitions=[],
        )
    if state["phase"] == "CLOSE":
        return _advance_result(state, advanced=False, next_step="STOP", transitions=[])
    if state["authorization_state"] not in {
        "MAINTAINER_AUTHORIZED",
        "IMPLEMENTATION_READY",
        "IMPLEMENTING",
        "VERIFIED",
    }:
        return _advance_result(
            state,
            advanced=False,
            next_step="STOP_FOR_MAINTAINER_AUTHORIZATION",
            transitions=[],
        )
    if any(worker.get("status") == "SPAWN_UNKNOWN" for worker in state["workers"]):
        return _advance_result(
            state, advanced=False, next_step="RECONCILE_AMBIGUOUS_SPAWN", transitions=[]
        )
    key = record.get("task_key")
    task = state["tasks"].get(key)
    if task is None:
        raise WorkflowError("advance requires a registered task_key")
    if state["phase"] == "DISPATCH":
        existing = next(
            (
                worker
                for worker in state["workers"]
                if worker.get("task_key") == key
                and worker.get("status") in ACTIVE_WORKER_STATES
            ),
            None,
        )
        if existing is not None:
            if existing.get("execution_mode") == "root":
                return _advance_result(
                    state,
                    advanced=False,
                    next_step="ROOT_IMPLEMENTATION_IN_PROGRESS",
                    transitions=[],
                )
            return _advance_result(
                state,
                advanced=False,
                next_step="AWAIT_MANUAL_WORKER_REFERENCE",
                transitions=[],
            )
        if record.get("writer_mode") != "subagent":
            raise WorkflowError(
                "manual writer continuation requires writer_mode subagent"
            )
        _create_dispatch_intent(state, task, record)
        return _advance_result(
            state,
            advanced=True,
            next_step="AWAIT_MANUAL_WORKER_REFERENCE",
            transitions=[],
            manual_launch_packet={
                "task_key": key,
                "owned_paths": task["owned_paths"],
                "selected_route": task["route"],
                "manual_adapter": True,
            },
        )
    transitions: list[str] = []
    while state["phase"] in {"RECONCILE", "DELIBERATE", "DECIDE"}:
        target = next(iter(ALLOWED_TRANSITIONS[state["phase"]]))
        transition(state, target)
        transitions.append(target)

    if state["phase"] == "PLAN":
        if state["authorization_state"] != "IMPLEMENTATION_READY":
            return _advance_result(
                state,
                advanced=False,
                next_step="ACQUIRE_REQUIRED_OWNERSHIP",
                transitions=transitions,
            )
        writer_mode = record.get("writer_mode", "root")
        if writer_mode == "root":
            root_writer = _activate_root_writer(state, task, record)
        elif writer_mode == "subagent":
            _create_dispatch_intent(state, task, record)
            root_writer = None
        else:
            raise WorkflowError("writer_mode must be root or subagent")
        transition(state, "DISPATCH")
        state["authorization_state"] = "IMPLEMENTING"
        transitions.append("DISPATCH")
        if root_writer is not None:
            return _advance_result(
                state,
                advanced=True,
                next_step="ROOT_IMPLEMENTATION_IN_PROGRESS",
                transitions=transitions,
                root_writer={
                    "reference": root_writer["reference"],
                    "attempt": root_writer["attempt"],
                },
            )
        return _advance_result(
            state,
            advanced=True,
            next_step="AWAIT_MANUAL_WORKER_REFERENCE",
            transitions=transitions,
            manual_launch_packet={
                "task_key": key,
                "owned_paths": task["owned_paths"],
                "selected_route": task["route"],
                "manual_adapter": True,
            },
        )

    if state["phase"] != "VERIFY":
        raise WorkflowError("advance reached an unsupported phase")
    handoff = _latest_handoff(state, key)
    if handoff is None:
        return _advance_result(
            state, advanced=False, next_step="AWAIT_HANDOFF", transitions=transitions
        )
    subject = _as_object(handoff.get("identity"), "handoff identity")
    requested_verification = record.get("verification_request")
    reusable_verification = _reusable_verification(
        state, key, subject, requested_verification
    )
    if reusable_verification is None:
        return _advance_result(
            state,
            advanced=False,
            next_step="RUN_OR_RECORD_REQUIRED_VERIFICATION",
            transitions=transitions,
            verification_subject=copy.deepcopy(subject),
        )
    if state["authorization_state"] != "VERIFIED":
        return _advance_result(
            state,
            advanced=False,
            next_step="RUN_OR_RECORD_REQUIRED_VERIFICATION",
            transitions=transitions,
        )
    if not _matching_handoff_verification(handoff, reusable_verification):
        return _advance_result(
            state,
            advanced=False,
            next_step="RUN_OR_RECORD_REQUIRED_VERIFICATION",
            transitions=transitions,
        )
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
            return _advance_result(
                state,
                advanced=False,
                next_step="AWAIT_COLD_REVIEW",
                transitions=transitions,
                review_packet=copy.deepcopy(review),
            )
        if review["status"] == "REPAIR_IN_PROGRESS" and review["subject"] != subject:
            review["subject"] = copy.deepcopy(subject)
            review["status"] = "REQUIRED"
        review_status = record.get("review_status")
        if review_status is not None:
            if review_status == "PASS":
                review["status"] = "PASS"
                review["completed_sequence"] = _next_sequence(state)
            elif review_status == "REPAIR_REQUIRED":
                review["repair_cycles"] += 1
                review["status"] = "REPAIR_IN_PROGRESS"
                _transition_with_authorization(state, "PLAN", "MAINTAINER_AUTHORIZED")
                transitions.append("PLAN")
                return _advance_result(
                    state,
                    advanced=True,
                    next_step="REPAIR_IN_SCOPE",
                    transitions=transitions,
                    review_packet=copy.deepcopy(review),
                )
            else:
                raise WorkflowError("review_status must be PASS or REPAIR_REQUIRED")
        if review["status"] != "PASS":
            return _advance_result(
                state,
                advanced=False,
                next_step="AWAIT_COLD_REVIEW",
                transitions=transitions,
                review_packet=copy.deepcopy(review),
            )
    return _advance_result(
        state,
        advanced=False,
        next_step="AWAIT_MAINTAINER_INTEGRATION_AUTHORIZATION",
        transitions=transitions,
        integration_evidence=_integration_evidence(
            key, handoff, reusable_verification, review
        ),
    )


def load_state(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    return validate_state(state)


def write_state(path: str, state: dict[str, Any]) -> None:
    validate_state(state)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temporary = tempfile.mkstemp(
        prefix=".temper-workflow-", suffix=".json", dir=directory
    )
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
    parser.add_argument(
        "command",
        choices=[
            "status",
            "validate",
            "transition",
            "authorize",
            "validate-task",
            "register-worker",
            "reconcile-worker",
            "monitor-worker",
            "acquire-ownership",
            "release-ownership",
            "record-evidence",
            "validate-route-trial",
            "record-route-trial",
            "record-verification",
            "verification-reuse",
            "checkpoint",
            "compile-context",
            "ingest-handoff",
            "review-required",
            "next-action",
            "advance",
        ],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = load_state(args.state)
        record = _record_argument(args.record)
        changed = False
        if args.command == "status":
            result = {
                "phase": state["phase"],
                "authorization_state": state["authorization_state"],
                "next_action": next_action(state),
            }
        elif args.command == "validate":
            result = {"valid": True}
        elif args.command == "transition":
            result, changed = transition(state, record.get("target")), True
        elif args.command == "authorize":
            result, changed = authorize(state, record), True
        elif args.command == "validate-task":
            result = {
                "valid": True,
                "task": validate_task(
                    record, exact_base=state["repository"]["exact_base"]
                ),
            }
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
        elif args.command == "validate-route-trial":
            trial = validate_route_trial(record)
            result = {
                "valid": True,
                "trial_id": trial["trial_id"],
                "eligible_for_default_route_decision": trial[
                    "eligible_for_default_route_decision"
                ],
            }
        elif args.command == "record-route-trial":
            result, changed = record_route_trial(state, record), True
        elif args.command == "record-verification":
            result, changed = record_verification(state, record), True
        elif args.command == "verification-reuse":
            result = verification_reuse(
                record.get("verification"),
                record.get("subject"),
                record.get("requested"),
            )
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
