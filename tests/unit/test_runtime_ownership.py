from pathlib import Path

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.ownership import (
    RunOwnershipError,
    claim_abandoned_run_ownership,
    claim_released_run_ownership,
    claim_run_ownership,
    iter_run_ownership_claims,
    reconcile_run_ownership,
    released_run_claim_identity,
)


CLAIM = ContentIdentity("sha256", "4" * 64)


def test_run_ownership_is_exclusive_and_reopens_only_after_resolution(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()

    with claim_run_ownership(root, "run-owned", CLAIM) as first:
        with pytest.raises(RunOwnershipError, match="run_ownership_unavailable"):
            with claim_run_ownership(root, "run-owned", CLAIM):
                raise AssertionError("a competing owner acquired the run")
        first.resolve()

    with claim_run_ownership(root, "run-owned", CLAIM) as reconciled:
        reconciled.resolve()


def test_run_ownership_stays_blocked_when_terminal_resolution_is_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()

    with claim_run_ownership(root, "run-unresolved", CLAIM):
        pass

    with pytest.raises(RunOwnershipError, match="run_ownership_unresolved"):
        released_run_claim_identity(root, "run-unresolved")
    with pytest.raises(RunOwnershipError, match="run_ownership_unresolved"):
        with claim_run_ownership(root, "run-unresolved", CLAIM):
            raise AssertionError("an unresolved claim was reacquired")


def test_exact_unresolved_claim_can_be_reconciled_after_terminal_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()

    with claim_run_ownership(root, "run-reconciled", CLAIM):
        pass

    assert reconcile_run_ownership(root, "run-reconciled", CLAIM) == CLAIM
    assert released_run_claim_identity(root, "run-reconciled") == CLAIM
    with claim_released_run_ownership(root, "run-reconciled", CLAIM):
        pass


def test_abandoned_claim_requires_an_existing_unresolved_exact_lease(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()

    with claim_run_ownership(root, "run-abandoned", CLAIM):
        pass

    with claim_abandoned_run_ownership(root, "run-abandoned", CLAIM) as abandoned:
        abandoned.resolve()

    assert released_run_claim_identity(root, "run-abandoned") == CLAIM
    with pytest.raises(RunOwnershipError, match="^run_ownership_resolved$"):
        with claim_abandoned_run_ownership(root, "run-abandoned", CLAIM):
            raise AssertionError("a released claim was treated as abandoned")


def test_released_claim_never_recreates_a_missing_lock(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    with claim_run_ownership(root, "run-released", CLAIM) as owned:
        owned.resolve()
    lock = root / "runtime-ownership" / "run-released" / "lease.lock"
    lock.unlink()
    claim = released_run_claim_identity(root, "run-released")

    with pytest.raises(RunOwnershipError, match="run_ownership_path_invalid"):
        with claim_released_run_ownership(root, "run-released", claim):
            raise AssertionError("a missing released lock was recreated")

    assert not lock.exists()


def test_startup_claim_inventory_distinguishes_unresolved_and_released(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    other = ContentIdentity("sha256", "5" * 64)
    with claim_run_ownership(
        root,
        "run-unresolved-inventory",
        CLAIM,
        request_id="request-unresolved-inventory",
        artifact_id="artifact-unresolved-inventory",
        attempt_number=2,
    ):
        pass
    with claim_run_ownership(root, "run-released-inventory", other) as released:
        released.resolve()

    claims = iter_run_ownership_claims(root)

    assert tuple(
        (claim.run_id, claim.claim_identity, claim.resolved) for claim in claims
    ) == (
        ("run-released-inventory", other, True),
        ("run-unresolved-inventory", CLAIM, False),
    )
    unresolved = claims[1]
    assert unresolved.request_id == "request-unresolved-inventory"
    assert unresolved.artifact_id == "artifact-unresolved-inventory"
    assert unresolved.attempt_number == 2


def test_startup_ignores_directory_only_claim_initialization_loss(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    incomplete = root / "runtime-ownership" / "run-directory-only"
    incomplete.mkdir(parents=True)

    assert iter_run_ownership_claims(root) == ()

    with claim_run_ownership(root, "run-directory-only", CLAIM) as ownership:
        ownership.resolve()
    assert released_run_claim_identity(root, "run-directory-only") == CLAIM


def test_startup_repairs_claim_published_before_lease_lock(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    with claim_run_ownership(root, "run-claim-only", CLAIM):
        pass
    lock = root / "runtime-ownership" / "run-claim-only" / "lease.lock"
    lock.unlink()

    claims = iter_run_ownership_claims(root)

    assert tuple(
        (claim.run_id, claim.claim_identity, claim.resolved) for claim in claims
    ) == (("run-claim-only", CLAIM, False),)
    assert lock.read_bytes() == b"\0"
    with claim_abandoned_run_ownership(root, "run-claim-only", CLAIM) as abandoned:
        abandoned.resolve()
    assert released_run_claim_identity(root, "run-claim-only") == CLAIM


@pytest.mark.parametrize("run_id", ["cleanup", "cleanup-quarantine", "replay"])
def test_inventory_does_not_hide_claims_named_like_legacy_controls(
    tmp_path: Path,
    run_id: str,
) -> None:
    root = tmp_path.resolve()
    with claim_run_ownership(root, run_id, CLAIM):
        pass

    claims = iter_run_ownership_claims(root)

    assert tuple(claim.run_id for claim in claims) == (run_id,)
