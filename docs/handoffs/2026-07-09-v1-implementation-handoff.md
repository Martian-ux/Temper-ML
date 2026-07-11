# Temper ML v1 implementation handoff

**Status:** Historical checkpoint from the 2026-07-09 pause; implementation
resumed with Slice 0 repository/gate baseline.

This handoff preserves the pause-state facts. It is not the current execution
status; the adopted governing roadmap is
`docs/superpowers/plans/2026-07-09-temper-ml-v1-execution-roadmap.md`.

The 2026-07-11 maintainer alignment is recorded in the governing architecture
and roadmap, not retroactively treated as pause-state fact. It makes focused
local adapter use and verified export explicit v1 scope, selects a Windows
host with a WSL2 Ubuntu ROCm worker as the first real-hardware topology, and
keeps native Windows execution secondary and capability-gated.

## Roadmap position

The governing execution roadmap is
`docs/superpowers/plans/2026-07-09-temper-ml-v1-execution-roadmap.md`.

Implementation should begin with Slice 0 (repository and gate baseline), then
continue with Slice 1 (canonical store and evidence services). Slice 5 is the
first end-to-end fixture-product milestone; it is not the next implementation
step.

## Worktree state to preserve

Pre-existing storage-hardening work is present in:

- `.gitattributes`
- `src/temper_ml/store/canonical_json.py`
- `src/temper_ml/store/write_once.py`
- `tests/fixtures/identity/project-policy-v1.json`
- `tests/unit/test_canonical_json.py`
- `tests/unit/test_write_once.py`

The roadmap, architecture document, and the untracked execution-roadmap file
also have pre-existing changes. Do not reset or overwrite these changes while
continuing the work.

An untracked `output/` directory is present. It is quarantined from this work:
do not inspect, stage, or use its contents. It is ignored to prevent accidental
tracking.

## Partial Slice 0 work started in this session

The following uncommitted changes were made before work was stopped:

- Expanded `README.md` with product boundaries, setup instructions, fixture
  conventions, hardware-test expectations, and public-safety guidance.
- Added `ruff` and `mypy` development dependencies and minimal configuration
  in `pyproject.toml`.
- Added `format`, `lint`, and `typecheck` commands to
  `scripts/temper-gate.py`, and included them in `maintenance`.
- Extended `tests/unit/test_temper_gate.py` for those gate commands.

Not completed:

- `uv.lock` has not been refreshed after adding the new development tools.
- No CI workflow was added.
- The complete gate has not been run after the partial Slice 0 changes.
- No new Slice 1 module was created in this session.

## Last known verification

Before the partial Slice 0 changes, the existing unit suite was reported as
passing with one expected symlink-related skip. The full gate could not run
because `uv` was unavailable on the machine PATH. Re-run verification after
refreshing dependencies rather than relying on that earlier result.

## Recommended continuation

1. Review the completed Slice 0 baseline before reconciling it with any
   user-owned dirty state.
2. Preserve the active canonical JSON and write-once changes while completing
   their tests.
3. Start Slice 1 with a small, standalone public-export redaction service,
   then implement the canonical store layout, typed envelopes/projection
   registry, verification, and hash-linked event stream.
4. Add the CLI inspection commands only after their underlying storage
   contracts are stable.

## Design constraints for Slice 1

- Keep canonical records immutable and public-safe.
- Use synthetic fixtures only; no hardware, network, or external service may
  be required by default tests.
- Treat incomplete temporary files as non-canonical evidence and fail closed
  on identity mismatches.
- Preserve append-only lifecycle evidence and make derived state rebuildable.
