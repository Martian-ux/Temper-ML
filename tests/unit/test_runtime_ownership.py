from pathlib import Path

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.ownership import (
    RunOwnershipError,
    claim_released_run_ownership,
    claim_run_ownership,
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
