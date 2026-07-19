# Temper ML v1 Slice 8 handoff

**Status:** Slice 8 implementation candidate; publication and integration
remain maintainer-gated.

**Base:** `7d1f9d66c5a8591de02f414c1dd84f88d4c2ad31`

**Branch:** `codex/slice-8`

**Task:** `tml-v1-slice-8`, classified as normal implementation and limited
to the adopted library-backed local adapter runtime.

This handoff does not authorize staging, commit, push, merge, release,
deployment, cleanup, Slice 9 retention execution, or later automated loops.

## Delivered runtime

- `LibraryBackend` is the narrow Temper-owned port for capability probing,
  LoRA training, checkpoints, heartbeat and cooperative control, evaluation
  inference, and focused or batch local-use inference.
- `TransformersPeftBackend` uses local-only PyTorch, Transformers, PEFT, and
  Accelerate APIs. `DeterministicLibraryBackend` exercises the same contract
  without ML libraries, hardware, network access, downloaded models, private
  data, or library-owned canonical identifiers.
- `WslWorkerBackend` freezes a private portable request and launches one
  explicitly configured WSL worker with a fixed no-shell argument vector. The
  worker has no canonical-store dependency and returns only typed durable
  messages, a subject-bound response, and content-identified staged bytes.
- Host-to-worker and worker-to-host manifests bind roles, byte counts, and
  SHA-256 identities. Receipts reconstruct the exact manifest they attest;
  missing, partial, replaced, reversed-direction, linked, or corrupted data
  cannot complete ingestion.
- One serialized controller validates ordered lifecycle evidence, one resource
  coordinator owns declared accelerator capacity, and one durable run-scoped
  OS lease closes the cross-service unused-run race. An unresolved claim blocks
  relaunch until identity-bound reconciliation instead of duplicating work.
- Checkpoints bind the frozen recipe and decoded optimizer step. Final-step
  checkpoints remain auditable but are non-resumable, preventing a timeout
  after the final save from executing an extra optimizer update.
- Training, evaluation, and local use bind private model/tokenizer sources to
  the verified Temper base-model reference and tokenizer identity. Backend
  capabilities and library versions are probed rather than trusted from a
  caller-supplied runtime identity.
- The normalized three-member adapter bundle preserves the existing run,
  artifact, availability, integrity, evaluation, selection, local-use, and
  export contracts. Export still makes no deployment-readiness claim.

## Architecture decision

GitHub issue 12 was checked before implementation. Slice 8 adopts its bounded
ownership insight—one serialized owner per live run and declared resource—via
small local ports and a durable lease. It does not add an actor framework,
distributed orchestration, hosted coordinator, provider surface, or rewrite of
the stable deterministic services and immutable records.

The detailed decision, protocol, failure model, Windows/WSL topology, and
capability rules are in
`docs/superpowers/plans/2026-07-18-slice-8-runtime-architecture.md`.

## Changed paths

- `src/temper_ml/runtime/protocol.py`, `controller.py`, and `ownership.py` own
  typed lifecycle messages, deterministic reconstruction, serialized resource
  leases, and durable cross-process run ownership.
- `src/temper_ml/runtime/staging.py`, `worker_port.py`, `worker_process.py`, and
  `wsl_backend.py` own the content-verified process and cross-OS boundary.
- `src/temper_ml/runtime/library_backend.py`, `library_adapter.py`, and
  `library_double.py` own the real library seam, normalized adapter service,
  inference runtime, and deterministic double.
- `src/temper_ml/app_services/runs.py`, `local_use.py`, and runtime integrity
  modules project verified boundary evidence into the existing canonical
  lifecycle without allowing the worker to write records.
- `pyproject.toml` and `uv.lock` add a separately installable `runtime`
  dependency extra; the normal deterministic gate does not require it.
- `tests/unit`, `tests/contract`, `tests/integration_fixture`, and
  `tests/hardware` cover protocol, ownership, staging, worker supervision,
  runtime/service compatibility, and an explicitly opt-in hardware smoke path.
- `README.md` documents configuration and non-capabilities without embedding a
  machine-specific value.

## Acceptance and verification evidence

The complete Slice 8 targeted suite passed on the assembled implementation:
49 tests passed and the hardware test skipped because it is explicitly
capability-gated. Ruff formatting and lint, mypy over `src`, bytecode
compilation, and lock consistency are separate required checks.

The exact final candidate must pass one full repository gate after the single
registered cold-review thread verifies the repaired bytes:

```text
python scripts/temper-gate.py --bootstrap-uv temp all
```

Review and verification evidence is reusable only when bound to the exact base,
candidate byte manifest, command semantics, environment and lock identity, and
relevant generated inputs. Hardware success is not required by the default
gate; a hardware run is truthful only when every exact local prerequisite is
explicitly enabled.

## Scope and safety

All repository fixtures and test content are synthetic. Canonical evidence
contains portable identities and sanitized capability facts, never absolute
paths, host or user names, process identifiers, private URLs, raw model bytes,
or worker diagnostics. Private request, worker, lease, staging, and hardware
configuration state remains ignored and outside the public candidate.

There is no open product decision in Slice 8. Publication and integration are
the remaining maintainer-only decisions. The candidate must not be staged or
integrated until exact-candidate review, the one final full gate, and the final
public-safety inspection all pass; recovery material must remain intact.
