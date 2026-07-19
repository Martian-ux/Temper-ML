from pathlib import Path

import pytest

from temper_ml.domain.projections import ContentIdentity
from temper_ml.runtime.ownership import RunOwnershipError, claim_run_ownership


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
