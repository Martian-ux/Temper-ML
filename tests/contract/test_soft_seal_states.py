from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.evaluations import EvaluationService
from temper_ml.domain.evaluations import (
    CaseSuiteKind,
    EvaluationCase,
    EvaluationSuite,
    EvaluatorKind,
    EvaluatorSpec,
    MetricDirection,
    SuiteEvidenceState,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.domain.records import record_reference


def _identity(label: str) -> ContentIdentity:
    return ContentIdentity("sha256", hashlib.sha256(label.encode()).hexdigest())


def _suite(suite_id: str) -> EvaluationSuite:
    return EvaluationSuite(
        suite_id,
        CaseSuiteKind.CONFIRMATION,
        SuiteEvidenceState.UNSEALED,
        (
            EvaluationCase("case-alpha", _identity(f"{suite_id}:case-alpha")),
            EvaluationCase("case-beta", _identity(f"{suite_id}:case-beta")),
        ),
        (
            EvaluatorSpec(
                "format-check",
                EvaluatorKind.FORMAT_CHECK,
                "format_validity",
                MetricDirection.MAXIMIZE,
            ),
        ),
    )


def test_inspection_changes_confirmation_evidence_state_and_is_not_resealable(
    tmp_path: Path,
) -> None:
    service = EvaluationService(tmp_path)
    initial = service.register_suite(_suite("suite-inspection"))
    sealed = service.seal_suite(initial)
    inspected = service.inspect_confirmation_suite(sealed)

    assert initial.state is SuiteEvidenceState.UNSEALED
    assert sealed.state is SuiteEvidenceState.SEALED
    assert inspected.state is SuiteEvidenceState.UNSEALED
    assert inspected.prior_suite is not None
    assert inspected.prior_suite.identity == sealed.identity
    assert inspected.case_membership_identity == sealed.case_membership_identity
    with pytest.raises(ApplicationServiceError, match="cannot_seal"):
        service.seal_suite(inspected)
    assert service.inspect_confirmation_suite(sealed) == inspected


def test_modification_contamination_and_retirement_are_immutable_revisions(
    tmp_path: Path,
) -> None:
    service = EvaluationService(tmp_path)
    sealed = service.seal_suite(service.register_suite(_suite("suite-lifecycle")))
    modified = service.modify_suite(
        sealed,
        cases=(
            *sealed.cases,
            EvaluationCase("case-gamma", _identity("suite-lifecycle:case-gamma")),
        ),
    )
    contaminated = service.mark_suite_contaminated(modified)
    with pytest.raises(ApplicationServiceError, match="suite_contaminated"):
        service.modify_suite(
            contaminated,
            cases=(
                *contaminated.cases,
                EvaluationCase("case-delta", _identity("suite-lifecycle:case-delta")),
            ),
        )
    retired = service.retire_suite(contaminated)

    assert modified.state is SuiteEvidenceState.MODIFIED
    assert modified.case_membership_identity != sealed.case_membership_identity
    assert contaminated.state is SuiteEvidenceState.CONTAMINATED
    assert retired.state is SuiteEvidenceState.RETIRED
    assert retired.prior_suite is not None
    assert retired.prior_suite.identity == contaminated.identity
    revisions = [
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, EvaluationSuite)
    ]
    assert {revision.state for revision in revisions} == {
        SuiteEvidenceState.UNSEALED,
        SuiteEvidenceState.SEALED,
        SuiteEvidenceState.MODIFIED,
        SuiteEvidenceState.CONTAMINATED,
        SuiteEvidenceState.RETIRED,
    }
    assert service.store.verify().record_counts["evaluation_suite"] == 5


def test_case_membership_identity_is_order_independent_but_content_bound() -> None:
    suite = _suite("suite-membership")
    reordered = EvaluationSuite(
        suite.suite_id,
        suite.kind,
        suite.state,
        tuple(reversed(suite.cases)),
        suite.evaluators,
    )
    changed = EvaluationSuite(
        suite.suite_id,
        suite.kind,
        suite.state,
        (
            EvaluationCase("case-alpha", _identity("changed-alpha")),
            suite.cases[1],
        ),
        suite.evaluators,
    )

    assert reordered.case_membership_identity == suite.case_membership_identity
    assert reordered.identity == suite.identity
    assert changed.case_membership_identity != suite.case_membership_identity
    assert changed.identity != suite.identity


def test_suite_transition_retry_recovers_exact_successor_and_missing_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = EvaluationService(tmp_path)
    initial = service.register_suite(_suite("suite-interrupted-transition"))

    def interrupt_after_record(*args, **kwargs):
        del args, kwargs
        raise ApplicationServiceError("simulated_interruption")

    monkeypatch.setattr(service, "_append", interrupt_after_record)
    with pytest.raises(ApplicationServiceError, match="simulated_interruption"):
        service.seal_suite(initial)

    restarted = EvaluationService(tmp_path)
    recovered = restarted.seal_suite(initial)
    expected = replace(
        initial,
        state=SuiteEvidenceState.SEALED,
        prior_suite=record_reference(initial),
    )
    transition_events = tuple(
        event
        for stream in restarted.store.iter_streams()
        for event in stream.events
        if event.event_type == "evaluation_suite_state_changed"
    )

    assert recovered == expected
    assert len(transition_events) == 1


def test_suite_lineage_rejects_forks_even_when_one_supplied_revision_is_a_leaf(
    tmp_path: Path,
) -> None:
    service = EvaluationService(tmp_path)
    initial = service.register_suite(_suite("suite-forked-lineage"))
    sealed = replace(
        initial,
        state=SuiteEvidenceState.SEALED,
        prior_suite=record_reference(initial),
    )
    modified = replace(
        initial,
        state=SuiteEvidenceState.MODIFIED,
        cases=(
            *initial.cases,
            EvaluationCase("case-fork", _identity("suite-forked-lineage:case-fork")),
        ),
        prior_suite=record_reference(initial),
    )
    service.store.write_record(sealed)
    service.store.write_record(modified)

    with pytest.raises(ApplicationServiceError, match="revision_conflict"):
        service.retire_suite(sealed)
