from pathlib import Path

import pytest

from temper_ml.app_services.errors import ApplicationServiceError
from temper_ml.app_services.fixture_journey import FixtureJourneyService
from temper_ml.app_services.retention import ByteClass, RetentionService


def test_inventory_rejects_a_link_without_reading_or_removing_its_target(
    tmp_path: Path,
) -> None:
    FixtureJourneyService(tmp_path).setup_project()
    outside = tmp_path / "outside-heavy-bytes.bin"
    outside.write_bytes(b"outside synthetic bytes")
    logs = tmp_path / ".temper-fixture-output" / "logs"
    logs.mkdir(parents=True)
    linked = logs / "linked-debug.bin"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")

    with pytest.raises(
        ApplicationServiceError, match="^storage_inventory_link_forbidden$"
    ):
        RetentionService(tmp_path).inventory()

    assert outside.read_bytes() == b"outside synthetic bytes"


def test_unknown_runtime_bytes_are_inventoried_but_never_cleanup_eligible(
    tmp_path: Path,
) -> None:
    FixtureJourneyService(tmp_path).setup_project()
    unknown = tmp_path / ".temper-fixture-output" / "unclassified" / "bytes.bin"
    unknown.parent.mkdir(parents=True)
    unknown.write_bytes(b"synthetic unknown bytes")
    service = RetentionService(tmp_path)
    entry = next(
        item
        for item in service.inventory().entries
        if item.byte_class is ByteClass.UNKNOWN
    )

    assert entry.deletable is False
    with pytest.raises(ApplicationServiceError, match="^cleanup_selection_protected$"):
        service.plan((entry.entry_id,))

    assert unknown.read_bytes() == b"synthetic unknown bytes"


def test_unrecorded_member_inside_a_known_artifact_directory_stays_protected(
    tmp_path: Path,
) -> None:
    journey = FixtureJourneyService(tmp_path)
    journey.setup_project()
    journey.import_dataset()
    journey.resolve_candidates()
    journey.launch_candidates()
    rogue = (
        tmp_path
        / ".temper-fixture-output"
        / "artifacts"
        / "artifact-fixture-runtime"
        / "unrecorded-member.bin"
    )
    rogue.write_bytes(b"synthetic unrecorded member")

    service = RetentionService(tmp_path)
    entry = next(
        item
        for item in service.inventory().entries
        if item.logical_key.endswith("unrecorded-member.bin")
    )

    assert entry.byte_class is ByteClass.UNKNOWN
    assert entry.deletable is False
    with pytest.raises(ApplicationServiceError, match="^cleanup_selection_protected$"):
        service.plan((entry.entry_id,))
    assert rogue.read_bytes() == b"synthetic unrecorded member"
