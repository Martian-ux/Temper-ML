from pathlib import Path

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.fixture_journey import (
    FixtureJourneyService,
    WorkspaceQueryService,
)
from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.fixture_inference import InferenceSettings


def test_fixture_journey_is_staged_and_restart_is_honest(tmp_path: Path) -> None:
    service = FixtureJourneyService(tmp_path)

    assert service.setup_project()["status"] == "open"
    imported = service.import_dataset()
    assert imported["statistics"]["accepted_rows"] == 3
    assert imported["private_preview"] is True

    restarted = WorkspaceQueryService(tmp_path).view()
    assert restarted["dataset"]["prepared_bytes_available"] is False
    assert restarted["dataset"]["reimport_required"] is True

    with pytest.raises(ApplicationServiceError, match="dataset_reimport_required"):
        FixtureJourneyService(tmp_path).resolve_candidates()


def test_fixture_journey_runs_two_real_candidates_and_gates_local_use(
    tmp_path: Path,
) -> None:
    service = FixtureJourneyService(tmp_path)
    service.setup_project()
    service.import_dataset()
    resolved = service.resolve_candidates()
    assert [item["key"] for item in resolved["candidates"]] == ["ember", "slate"]
    assert resolved["candidates"][0]["resolution"]["manifest"]["rank"] == 4
    assert resolved["candidates"][1]["resolution"]["manifest"]["rank"] == 8

    runs = service.launch_candidates()["runs"]
    assert len(runs) == 2
    assert all(run["verified_artifact"] for run in runs)

    comparison = service.compare(prompt="Rewrite the synthetic quality note")
    assert comparison["synchronized"] is True
    assert len(comparison["outputs"]) == 2
    assert comparison["outputs"][0]["output"] != comparison["outputs"][1]["output"]

    with pytest.raises(ApplicationServiceError, match="local_use_selection_required"):
        service.focused_local_use(candidate_key="ember", prompt="Before selection")

    review = service.record_solo_review(
        notes="Both outputs satisfy the declared fixture format.",
        ratings={"ember": 1, "slate": 1},
        declaration="I reviewed the synchronized prompt, settings, and outputs.",
    )
    assert review["mode"] == "solo"
    evaluation = service.evaluate_candidates()
    assert evaluation["recommendation"]["confidence"] == "low"
    assert "qualified_objective_tradeoff" in evaluation["recommendation"]["conflicts"]

    assert service.record_decision(candidate_key="ember")["status"] == "selected"
    focused = service.focused_local_use(candidate_key="ember", prompt="After selection")
    assert focused["focused_local_use"] is True
    assert focused["general_chat"] is False

    review_identity = ContentIdentity(
        review["review"]["identity"]["algorithm"],
        review["review"]["identity"]["value"],
    )
    captured = service.capture_review(review_identity)
    assert captured["suite_state"] == "modified"
    capture_identity = ContentIdentity(
        captured["case_content_identity"]["algorithm"],
        captured["case_content_identity"]["value"],
    )
    with_capture = service.focused_local_use(
        candidate_key="ember",
        prompt="Captured local use",
        capture_identity=capture_identity,
    )
    assert with_capture["saved_canonical_session"] is True


def test_fixture_journey_supports_blind_seal_before_reveal(tmp_path: Path) -> None:
    service = FixtureJourneyService(tmp_path)
    service.setup_project()
    service.import_dataset()
    service.resolve_candidates()
    service.launch_candidates()
    service.compare(prompt="Blind synthetic comparison")

    prepared = service.prepare_blind_review()
    packet = prepared["packet"]
    assert prepared["identities_revealed"] is False
    encoded = str(packet)
    assert "artifact-fixture-runtime" not in encoded
    aliases = [item["alias"] for item in packet["entries"][0]["outputs"]]

    sealed = service.seal_blind_review(
        notes="Both aliases satisfy the fixture format.",
        ratings={alias: 1 for alias in aliases},
        declaration="I judged the aliases before reveal.",
    )
    assert sealed["stage"] == "blind_sealed"
    assert sealed["identities_revealed"] is False
    revealed = service.reveal_blind_review()
    assert revealed["stage"] == "blind_revealed"
    assert revealed["identities_revealed"] is True


def test_evaluation_capture_uses_an_explicit_completed_review(tmp_path: Path) -> None:
    service = FixtureJourneyService(tmp_path)
    service.setup_project()
    service.import_dataset()
    service.resolve_candidates()
    service.launch_candidates()
    service.compare(prompt="Explicit review identity")

    first = service.record_solo_review(
        notes="First complete review.",
        ratings={"ember": 1, "slate": 1},
        declaration="I completed the first review.",
    )
    service.record_solo_review(
        notes="Second complete review.",
        ratings={"ember": 1, "slate": 1},
        declaration="I completed the second review.",
    )
    service.prepare_blind_review()

    first_identity = ContentIdentity(
        first["review"]["identity"]["algorithm"],
        first["review"]["identity"]["value"],
    )
    captured = service.capture_review(first_identity)
    assert captured["case_content_identity"] == first["review"]["identity"]

    prepared = next(
        review
        for review in service.workspace()["evaluation"]["reviews"]
        if review["stage"] == "blind_prepared"
    )
    prepared_identity = ContentIdentity(
        prepared["reference"]["identity"]["algorithm"],
        prepared["reference"]["identity"]["value"],
    )
    with pytest.raises(
        ApplicationServiceError, match="evaluation_capture_review_incomplete"
    ):
        service.capture_review(prepared_identity)


def test_workspace_rechecks_artifact_bytes_and_surfaces_failures(
    tmp_path: Path,
) -> None:
    service = FixtureJourneyService(tmp_path)
    service.setup_project()
    service.import_dataset()
    service.resolve_candidates()
    service.launch_candidates()

    artifacts = tmp_path / ".temper-fixture-output" / "artifacts"
    (artifacts / "artifact-fixture-runtime" / "adapter.bin").unlink()
    (artifacts / "artifact-fixture-challenger" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )

    by_key = {item["key"]: item for item in service.workspace()["artifacts"]}
    assert by_key["ember"]["available"] is False
    assert by_key["ember"]["integrity_status"] == "failed"
    assert by_key["ember"]["failure_code"] == "artifact_structure_mismatch"
    assert by_key["slate"]["available"] is False
    assert by_key["slate"]["integrity_status"] == "failed"
    assert by_key["slate"]["failure_code"] == "artifact_content_identity_mismatch"


def test_inference_settings_remain_bounded() -> None:
    assert InferenceSettings(0, 64, 17).to_dict() == {
        "temperature": 0,
        "maximum_tokens": 64,
        "seed": 17,
    }
