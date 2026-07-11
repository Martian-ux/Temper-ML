# Temper ML v1 foundation handoff

**Status:** Current handoff for the next fresh v1 implementation task
**Date:** 2026-07-11
**Branch:** `codex/v1-local-adapter-foundation`
**Pre-handoff implementation head:** `135b176cdfd4da73d409856608e8077f2c7c450b`

This handoff replaces conversation replay as the starting context for the next
v1 slice. The governing product contract is the architecture document at
`docs/superpowers/specs/2026-06-30-temper-ml-architecture-design.md`. Execution
order and slice acceptance are governed by
`docs/superpowers/plans/2026-07-09-temper-ml-v1-execution-roadmap.md`.

## Accepted product contract

- Temper v1 is a local-only LLM adapter experimentation product.
- Core v1 includes adapter training, deterministic evaluation, focused local
  interactive and batch use, reproducible iteration, retention, bounded search,
  compatible LoRA merging, verified export, and readiness assessment.
- Slices 0 through 10 are required v1 scope. Optional compatibility backends do
  not gate v1 completion.
- Temper owns manifests, lifecycle, evidence, artifacts, retention, and
  recommendations while mature libraries provide ML machinery.
- The first real-hardware target is a Windows host with a WSL2 Ubuntu ROCm
  worker. Native Windows execution is secondary and capability-gated behind the
  same runtime contract.
- No hosted Temper service, external trainer dashboard, model judge, general
  chat product, or deployment controller is part of core v1.

## Completed foundation

- Slice 0 repository, dependency-lock, cross-platform gate, fixture, and CI
  baseline.
- Strict canonical JSON encoding and reading, immutable write-once storage,
  projection identities, and path/symlink protections.
- Canonical store layout, hash-linked event stream, artifact and bundle
  identities, verification, recovery behavior, and adversarial tests.
- Product architecture and roadmap alignment for local adapter use and the
  Windows/WSL2 ROCm topology.
- Public-safety checks and repository-local GitHub noreply commit metadata.

## Verification at handoff

- The full repository gate passed on the pre-handoff implementation head:
  formatting, lint, typing, compilation, 48 passing unit tests, one expected
  Windows symlink skip, fixture help, and diff hygiene.
- The focused persistence adversarial suite passed 21 tests with two expected
  Windows symlink skips.
- The publishing task must rerun the full gate on the exact commit containing
  this handoff and report that result in the pull request.

## Next bounded work: finish Slice 1

The next fresh task should finish only the remaining canonical-store and
evidence-service scope:

1. Add typed record envelopes and the projection registry.
2. Add the redaction service and public-safe dump/export behavior.
3. Complete byte, record, and bundle verification integration.
4. Add CLI commands for status, dump, verify, and manifest inspection.
5. Add focused recovery, corruption, unsafe-path, redaction, and CLI tests.

Do not start project services, dataset behavior, UI, PyTorch or ROCm
dependencies, provider adapters, training, or evaluation in the Slice 1 task.

## Fresh-task protocol for every slice

1. Read `AGENTS.md`, the governing architecture, the execution roadmap, and the
   latest handoff before changing files.
2. Fetch the remote, verify the accepted prior-slice head, and require a clean
   worktree. Do not assume another pull request has merged.
3. Create one `codex/` branch for the slice from the explicitly accepted prior
   head. Keep one coherent implementation owner unless disjoint ownership is
   explicitly approved.
4. Bind the task to the slice's exact implementation list, proof, dependencies,
   non-goals, changed paths, and verification commands.
5. Use synthetic public fixtures. Keep hardware and network checks optional and
   capability-gated until their roadmap slice.
6. Run targeted checks while iterating and the full repository gate once on the
   final exact candidate.
7. Commit with repository-local GitHub noreply metadata, push without force,
   open a draft pull request, and write the next public-safe handoff.
8. Stop before merge, branch deletion, or worktree removal unless the maintainer
   explicitly authorizes that action.

## Integration note

This foundation branch is a review candidate, not an integration decision.
The accepted workflow-control pull request is now integrated on `main`. The
publication merge brought that history into this branch while retaining the
expanded quality-gate commands and their deduplicated execution behavior.
Future tasks must still fetch and verify the then-current `main`; do not
silently rebase, rewrite, or discard either product or workflow evidence.
