from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest

import temper_ml.app_services.evaluations as evaluation_services
from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.evaluations import (
    BLIND_MAPPING_PROJECTION,
    BlindCandidateOutput,
    BlindReviewInput,
    BlindReviewJudgment,
    EvaluationService,
)
from temper_ml.cli import main
from temper_ml.domain.artifacts import Artifact
from temper_ml.domain.evaluations import (
    ArtifactIntegrityStatus,
    EvaluationResult,
    EvidenceStatus,
    Review,
    ReviewCandidate,
    ReviewEntry,
    ReviewMode,
    ReviewRating,
    ReviewStage,
)
from temper_ml.domain.projections import content_identity
from temper_ml.domain.records import record_reference
from temper_ml.domain.runs import EvaluationMode
from temper_ml.store.canonical_json import dumps_canonical_json
from temper_ml.store.evidence import TypedEvidenceStore


def _artifacts(root: Path, capsys) -> tuple[Artifact, Artifact]:
    assert main(["fixture-workflow", str(root)]) == 0
    capsys.readouterr()
    store = TypedEvidenceStore(root)
    first = next(
        stored.record
        for stored in store.iter_records()
        if isinstance(stored.record, Artifact)
    )
    second = replace(first, artifact_id="artifact-blind-alternative")
    store.write_record(second)
    return first, second


def _inputs(first: Artifact, second: Artifact) -> tuple[BlindReviewInput, ...]:
    return (
        BlindReviewInput(
            "prompt-alpha",
            {"text": "Rewrite the synthetic phrase."},
            {"temperature": 0, "maximum_tokens": 32},
            (
                BlindCandidateOutput(
                    record_reference(first),
                    {"text": "Synthetic response alpha one"},
                ),
                BlindCandidateOutput(
                    record_reference(second),
                    {"text": "Synthetic response alpha two"},
                ),
            ),
        ),
        BlindReviewInput(
            "prompt-beta",
            {"text": "Format the synthetic value."},
            {"temperature": 0, "maximum_tokens": 32},
            (
                BlindCandidateOutput(
                    record_reference(second),
                    {"text": "Synthetic response beta two"},
                ),
                BlindCandidateOutput(
                    record_reference(first),
                    {"text": "Synthetic response beta one"},
                ),
            ),
        ),
    )


def _judgments(packet) -> tuple[BlindReviewJudgment, ...]:
    return tuple(
        BlindReviewJudgment(
            entry.prompt_id,
            "Both synthetic outputs were reviewed before identity reveal.",
            tuple(
                ReviewRating(alias, "task_fit", 1 if index == 0 else 0)
                for index, alias in enumerate(packet.aliases)
            ),
        )
        for entry in packet.entries
    )


def test_optional_blind_review_persists_hidden_mapping_and_restarts_before_reveal(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    class ReverseSystemRandom:
        def shuffle(self, values) -> None:
            values.reverse()

    hiding_nonce = "a" * 64
    monkeypatch.setattr(
        evaluation_services.secrets,
        "SystemRandom",
        lambda: ReverseSystemRandom(),
    )
    monkeypatch.setattr(
        evaluation_services.secrets,
        "token_hex",
        lambda size: hiding_nonce if size == 32 else "",
    )
    first, second = _artifacts(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    packet = service.prepare_blind_review(
        "review-blind-synthetic",
        _inputs(first, second),
    )
    public_packet = dumps_canonical_json(packet.public_fields())
    preparation = next(
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, Review)
        and stored.record.stage is ReviewStage.BLIND_PREPARED
    )

    assert packet.leak_audit_passed is True
    assert packet.aliases == ("candidate-001", "candidate-002")
    assert not hasattr(packet, "candidate_mappings")
    assert not hasattr(packet, "hiding_nonce")
    assert first.artifact_id.encode() not in public_packet
    assert second.artifact_id.encode() not in public_packet
    assert first.identity.value.encode() not in public_packet
    assert second.identity.value.encode() not in public_packet
    assert hiding_nonce.encode() not in public_packet
    expected_randomized = tuple(
        reversed(
            sorted(
                (record_reference(first), record_reference(second)),
                key=lambda reference: (
                    reference.record_type,
                    reference.logical_id,
                    reference.identity.value,
                ),
            )
        )
    )
    assert (
        tuple(mapping.candidate for mapping in preparation.candidate_mappings)
        == expected_randomized
    )
    guessed_commitments = {
        content_identity(
            BLIND_MAPPING_PROJECTION,
            {
                "mappings": [mapping.to_dict() for mapping in guessed_mapping],
            },
        )
        for ordering in permutations(
            (record_reference(first), record_reference(second))
        )
        for guessed_mapping in (
            tuple(
                ReviewCandidate(alias, candidate)
                for alias, candidate in zip(packet.aliases, ordering, strict=True)
            ),
        )
    }
    assert packet.mapping_commitment not in guessed_commitments
    assert preparation.hiding_nonce == hiding_nonce
    assert preparation.packet_identity == packet.packet_identity
    with pytest.raises(ApplicationServiceError, match="identity_not_revealed"):
        service.record_result(
            EvaluationResult(
                "result-blind-prepared-invalid",
                record_reference(first),
                EvaluationMode.LIGHT_EVALUATION,
                ArtifactIntegrityStatus.PASSED,
                first.integrity_evidence,
                EvidenceStatus.SUBJECTIVE_ONLY,
                review=record_reference(preparation),
            )
        )

    judgments = _judgments(packet)
    sealed = EvaluationService(tmp_path).seal_blind_review(
        "review-blind-synthetic",
        packet,
        judgments,
        reviewer_declaration=(
            "I judged every alias before candidate identities were revealed."
        ),
    )
    sealed_bytes = dumps_canonical_json(sealed.to_dict())

    assert sealed.stage is ReviewStage.BLIND_SEALED
    assert sealed.prior_review == record_reference(preparation)
    assert sealed.candidate_mappings == ()
    assert first.artifact_id.encode() not in sealed_bytes
    assert second.artifact_id.encode() not in sealed_bytes
    restarted = EvaluationService(tmp_path)
    revealed = restarted.reveal_blind_review(sealed, packet)
    assert revealed.stage is ReviewStage.BLIND_REVEALED
    assert revealed.prior_review == record_reference(sealed)
    assert {mapping.candidate for mapping in revealed.candidate_mappings} == {
        record_reference(first),
        record_reference(second),
    }

    subjective = restarted.record_result(
        EvaluationResult(
            "result-blind-subjective",
            record_reference(first),
            EvaluationMode.LIGHT_EVALUATION,
            ArtifactIntegrityStatus.PASSED,
            first.integrity_evidence,
            EvidenceStatus.SUBJECTIVE_ONLY,
            review=record_reference(revealed),
        )
    )
    assert subjective.evidence_status is EvidenceStatus.SUBJECTIVE_ONLY
    assert subjective.review == record_reference(revealed)
    assert restarted.store.verify().record_counts["review"] == 3
    public_dump = dumps_canonical_json(restarted.store.public_dump().value)
    assert hiding_nonce.encode() not in public_dump
    assert first.artifact_id.encode() not in public_dump
    assert second.artifact_id.encode() not in public_dump


def test_blind_review_transition_retries_recover_records_and_missing_events(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    first, second = _artifacts(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    packet = service.prepare_blind_review(
        "review-blind-interrupted",
        _inputs(first, second),
    )

    def interrupt_after_record(*args, **kwargs):
        del args, kwargs
        raise ApplicationServiceError("simulated_interruption")

    monkeypatch.setattr(service, "_append", interrupt_after_record)
    with pytest.raises(ApplicationServiceError, match="simulated_interruption"):
        service.seal_blind_review(
            packet.packet_id,
            packet,
            _judgments(packet),
            reviewer_declaration=(
                "I judged every alias before candidate identities were revealed."
            ),
        )

    restarted = EvaluationService(tmp_path)
    sealed = restarted.seal_blind_review(
        packet.packet_id,
        packet,
        _judgments(packet),
        reviewer_declaration=(
            "I judged every alias before candidate identities were revealed."
        ),
    )
    sealed_events = tuple(
        event
        for stream in restarted.store.iter_streams()
        for event in stream.events
        if event.event_type == "blind_review_judgment_sealed"
    )
    assert len(sealed_events) == 1

    monkeypatch.setattr(restarted, "_append", interrupt_after_record)
    with pytest.raises(ApplicationServiceError, match="simulated_interruption"):
        restarted.reveal_blind_review(sealed, packet)

    final_service = EvaluationService(tmp_path)
    revealed = final_service.reveal_blind_review(sealed, packet)
    reveal_events = tuple(
        event
        for stream in final_service.store.iter_streams()
        for event in stream.events
        if event.event_type == "blind_review_identities_revealed"
    )
    assert revealed.stage is ReviewStage.BLIND_REVEALED
    assert len(reveal_events) == 1


def test_blind_reveal_recomputes_sealed_payload_identity_from_committed_mapping(
    tmp_path: Path,
    capsys,
) -> None:
    first, second = _artifacts(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    packet = service.prepare_blind_review(
        "review-blind-sealed-payload-tamper",
        _inputs(first, second),
    )
    preparation = next(
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, Review)
        and stored.record.review_id == packet.packet_id
        and stored.record.stage is ReviewStage.BLIND_PREPARED
    )
    judgments = {item.prompt_id: item for item in _judgments(packet)}
    entries = tuple(
        ReviewEntry(
            entry.prompt_id,
            entry.prompt,
            entry.settings,
            entry.outputs,
            judgments[entry.prompt_id].notes,
            judgments[entry.prompt_id].ratings,
        )
        for entry in packet.entries
    )
    tampered_entries = (
        replace(
            entries[0],
            prompt={"text": "Altered synthetic prompt after packet commitment."},
        ),
        *entries[1:],
    )
    tampered_sealed = Review(
        review_id=packet.packet_id,
        mode=ReviewMode.BLIND,
        stage=ReviewStage.BLIND_SEALED,
        entries=tampered_entries,
        reviewer_declaration=(
            "I judged every alias before candidate identities were revealed."
        ),
        candidate_mappings=(),
        leak_audit_passed=True,
        packet_identity=packet.packet_identity,
        prior_review=record_reference(preparation),
    )
    service.store.write_record(tampered_sealed)

    with pytest.raises(ApplicationServiceError, match="review_revision_conflict"):
        service.reveal_blind_review(tampered_sealed, packet)


@pytest.mark.parametrize("tamper", ("reviewer_declaration", "candidate_mappings"))
def test_blind_reveal_rejects_mutated_sealed_lineage_fields(
    tmp_path: Path,
    capsys,
    tamper: str,
) -> None:
    first, second = _artifacts(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    packet = service.prepare_blind_review(
        f"review-blind-lineage-{tamper}",
        _inputs(first, second),
    )
    sealed = service.seal_blind_review(
        packet.packet_id,
        packet,
        _judgments(packet),
        reviewer_declaration=(
            "I judged every alias before candidate identities were revealed."
        ),
    )
    preparation = next(
        stored.record
        for stored in service.store.iter_records()
        if isinstance(stored.record, Review)
        and stored.record.review_id == packet.packet_id
        and stored.record.stage is ReviewStage.BLIND_PREPARED
    )
    candidate_mappings = preparation.candidate_mappings
    reviewer_declaration = sealed.reviewer_declaration
    if tamper == "reviewer_declaration":
        reviewer_declaration = "Altered declaration after judgment sealing."
    else:
        candidates = tuple(mapping.candidate for mapping in candidate_mappings)
        candidate_mappings = tuple(
            ReviewCandidate(mapping.alias, candidate)
            for mapping, candidate in zip(
                candidate_mappings,
                reversed(candidates),
                strict=True,
            )
        )
    tampered_reveal = replace(
        sealed,
        stage=ReviewStage.BLIND_REVEALED,
        reviewer_declaration=reviewer_declaration,
        candidate_mappings=candidate_mappings,
        prior_review=record_reference(sealed),
    )
    service.store.write_record(tampered_reveal)

    with pytest.raises(ApplicationServiceError, match="review_revision_conflict"):
        service.reveal_blind_review(sealed, packet)


def test_blind_review_lineage_rejects_reveal_forks(
    tmp_path: Path,
    capsys,
) -> None:
    first, second = _artifacts(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    packet = service.prepare_blind_review(
        "review-blind-forked",
        _inputs(first, second),
    )
    sealed = service.seal_blind_review(
        packet.packet_id,
        packet,
        _judgments(packet),
        reviewer_declaration=(
            "I judged every alias before candidate identities were revealed."
        ),
    )
    first_mapping = (
        ReviewCandidate(packet.aliases[0], record_reference(first)),
        ReviewCandidate(packet.aliases[1], record_reference(second)),
    )
    second_mapping = (
        ReviewCandidate(packet.aliases[0], record_reference(second)),
        ReviewCandidate(packet.aliases[1], record_reference(first)),
    )
    service.store.write_record(
        replace(
            sealed,
            stage=ReviewStage.BLIND_REVEALED,
            candidate_mappings=first_mapping,
            prior_review=record_reference(sealed),
        )
    )
    service.store.write_record(
        replace(
            sealed,
            stage=ReviewStage.BLIND_REVEALED,
            candidate_mappings=second_mapping,
            prior_review=record_reference(sealed),
        )
    )

    with pytest.raises(ApplicationServiceError, match="revision_conflict"):
        service.reveal_blind_review(sealed, packet)


def test_blind_packet_leak_audit_rejects_candidate_identifiers(
    tmp_path: Path,
    capsys,
) -> None:
    first, second = _artifacts(tmp_path, capsys)
    service = EvaluationService(tmp_path)
    leaky = BlindReviewInput(
        "prompt-leaky",
        {"text": "Synthetic prompt"},
        {"temperature": 0},
        (
            BlindCandidateOutput(
                record_reference(first),
                {"text": f"Candidate {first.artifact_id}"},
            ),
            BlindCandidateOutput(
                record_reference(second),
                {"text": "Synthetic neutral output"},
            ),
        ),
    )

    with pytest.raises(ApplicationServiceError, match="leak_detected"):
        service.prepare_blind_review("packet-leaky", (leaky,))
    assert not any(
        isinstance(stored.record, Review) for stored in service.store.iter_records()
    )


@pytest.mark.parametrize(
    ("prompt", "settings", "output", "private_marker"),
    (
        (
            {"api_token": "fixture-sensitive-value"},
            {"temperature": 0},
            {"text": "Synthetic neutral output"},
            "api_token",
        ),
        (
            {"text": "Synthetic prompt"},
            {"contact": "reviewer@example.invalid"},
            {"text": "Synthetic neutral output"},
            "reviewer@example.invalid",
        ),
        (
            {"text": "Synthetic prompt"},
            {"temperature": 0},
            {"text": "C:\\fixture-private\\review-output.txt"},
            "fixture-private",
        ),
        (
            {"text": "Synthetic prompt"},
            {"temperature": 0},
            {"text": "https://private.invalid/review-output"},
            "private.invalid",
        ),
        (
            {"text": "fixture-local-user"},
            {"temperature": 0},
            {"text": "Synthetic neutral output"},
            "fixture-local-user",
        ),
        (
            {"text": "Synthetic prompt"},
            {"label": "fixture-local-host"},
            {"text": "Synthetic neutral output"},
            "fixture-local-host",
        ),
    ),
)
def test_blind_packet_canonical_admission_rejects_private_fields_without_echo(
    tmp_path: Path,
    capsys,
    monkeypatch,
    prompt,
    settings,
    output,
    private_marker: str,
) -> None:
    first, second = _artifacts(tmp_path, capsys)
    monkeypatch.setattr(
        evaluation_services.RedactionContext,
        "current",
        classmethod(
            lambda cls: cls(
                local_usernames=("fixture-local-user",),
                local_hostnames=("fixture-local-host",),
            )
        ),
    )
    unsafe = BlindReviewInput(
        "prompt-private-admission",
        prompt,
        settings,
        (
            BlindCandidateOutput(record_reference(first), output),
            BlindCandidateOutput(
                record_reference(second),
                {"text": "Synthetic neutral alternative"},
            ),
        ),
    )
    service = EvaluationService(tmp_path)

    with pytest.raises(ApplicationServiceError) as error:
        service.prepare_blind_review("review-private-admission", (unsafe,))

    assert str(error.value) == "blind_review_packet_public_safety_failed"
    assert private_marker not in str(error.value)
    assert not any(
        isinstance(stored.record, Review) for stored in service.store.iter_records()
    )
