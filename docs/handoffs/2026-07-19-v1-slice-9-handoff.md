# Temper ML v1 Slice 9 handoff

**Status:** Slice 9 implementation candidate. Draft publication remains pending
final verification and public-safety prerequisites; merge and integration
remain maintainer-gated.

**Base:** `44c5827587ec1addce6d741700e50ba88fbf86f7`

**Branch:** `codex/slice-9`

This handoff does not authorize merge, release, deployment, canonical-record
deletion, automatic garbage collection, destructive repository cleanup, or
Slice 10 optimization loops.

## Delivered behavior

- Full retention is the default. Cleanup is confined to verified regular files
  under the fixed project-local heavy-byte root; `.temper`, unknown bytes,
  active staging, links, reparse points, and non-regular files fail closed.
- Inventory distinguishes logical, physical, and actually reclaimable bytes.
  Shared file objects count once physically and are reclaimable only when every
  known reference is selected.
- Cleanup preview freezes the exact inventory and selection, then reports
  retained and removed byte classes, affected canonical subjects, and separate
  consequences for resumability, inspectability, final artifacts, cache
  convenience, and debugging evidence.
- Confirmed execution uses a unique attempt identity, persists the complete
  portable intent, and holds one nonblocking project cleanup lease across plan
  revalidation, deletion, receipt assembly, and restart reconciliation. The
  lease prevents distinct attempts from forking cleanup or availability
  evidence. Before each unlink the executor fences affected artifact/checkpoint
  availability and revalidates the plan, path, metadata, link state, byte count,
  and content identity. It stops on the first failure and records a completed,
  partial, or failed immutable receipt. Restart reconciliation completes missing
  evidence without merging a later identical byte selection into the earlier
  execution.
- Strict replay launches a distinct new run only for the unchanged frozen
  experiment manifest. Adapted replay requires a persisted derived experiment,
  exact manifest diff, and explicit adapted-reproduction lineage.
- The CLI and local dashboard expose storage inventory, consequence-first
  cleanup, immutable results, replay readiness, and manifest differences while
  leaving cleanup and replay policy in application services.

The detailed boundary, protocol, failure model, and accepted non-capabilities
are in
`docs/superpowers/plans/2026-07-19-slice-9-retention-architecture.md`.

## Changed paths

- `src/temper_ml/domain/retention.py`, record registration, schemas, and public
  projection define the canonical cleanup receipt and keep it default-deny
  until explicitly projected.
- `src/temper_ml/app_services/retention.py` owns inventory, planning,
  consequence calculation, revalidation, bounded deletion, availability
  supersession, and receipt creation.
- `src/temper_ml/app_services/reproduction.py` owns strict and adapted replay
  validation and execution; fixture journey projections preserve the distinction
  across restart.
- `src/temper_ml/cli.py` and `src/temper_ml/ui` expose the operator workflows
  without accepting an arbitrary cleanup path.
- `tests/unit`, `tests/adversarial`, and `tests/integration_fixture` cover
  shared references, stale plans, partial failure, link rejection, unknown-byte
  protection, restart behavior, post-delete event/availability/receipt failure,
  distinct identical-byte executions, competing-execution serialization,
  cleanup availability, and strict/adapted replay.
- `README.md` documents the bounded commands and retention guarantees.

## Acceptance state

- Focused unit, adversarial, contract, integration-fixture, CLI, and UI coverage
  exercises the delivered retention and reproduction boundaries.
- Fault coverage includes interruption recovery, post-delete persistence
  failures, ambiguous removal, repeated identical bytes, and competing cleanup
  execution.
- Independent review completed on the assembled implementation with no open
  implementation finding.
- Draft publication remains pending the complete repository gate and final
  public-safety inspection on unchanged candidate bytes.

## Decisions and compatibility

Cleanup never deletes canonical records or manifests. Removed artifact members
and checkpoints are represented by superseding observations instead of
rewriting history. Strict replay preserves manifest identity; any changed
requirement produces a new derived experiment and is labeled adapted. These
decisions preserve the existing run, artifact, evaluation, recommendation,
local-use, export, and public-dump contracts.

## Scope, safety, and integration guidance

All tests and fixtures use synthetic project-local bytes. Public records and
views contain portable logical keys and content identities, never absolute
paths, user or host names, device/inode values, process identifiers, private
URLs, raw artifact bytes, or operational diagnostics. Non-public workflow
records remain ignored and outside the candidate.

No in-scope implementation defect is known at assembly. The draft PR must stop
before merge so the maintainer can make the separate integration decision on
the exact passing head.
