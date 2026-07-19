# Temper ML

Temper ML is a local-only product for LLM adapter experimentation. It is being
built to help one semi-technical user create, evaluate, compare, reproduce, use,
and retain adapter experiments without making an external trainer, dashboard,
or cloud account the source of truth.

## Product boundary

Temper owns project identity, dataset identity, resolved manifests, run
lifecycle evidence, artifact verification, retention state, and selection
evidence. Training libraries may provide implementation machinery, but their
files and identifiers are not canonical Temper records.

v1 is intentionally limited to LLM adapter experimentation. It is not a
general machine-learning platform, a full-model training system, a deployment
controller, or a general chat client. The approved product architecture is in
`docs/superpowers/specs/2026-06-30-temper-ml-architecture-design.md`; the
adopted execution sequence is in
`docs/superpowers/plans/2026-07-09-temper-ml-v1-execution-roadmap.md`.

Using an adapter is part of v1: a verified adapter can be loaded with its
compatible base model for focused local interactive or batch inference, saved
with its settings and provenance, and exported as a verified local bundle.
Temper does not turn that workflow into hosted serving or a general chat
product.

## Local execution target

Temper's core workflows are local-only. The initial real-hardware topology is a
Windows host for the launcher, loopback UI and CLI, application services, and
canonical project store, with an explicit WSL2 Ubuntu ROCm worker for training,
evaluation inference, and local adapter use on a supported AMD GPU.

The WSL worker receives immutable runtime requests and returns explicit
artifacts and evidence for ingestion; it does not become a second source of
truth. Native Windows PyTorch/ROCm execution may use the same runtime contract
when capability detection proves the required combination is supported. No
core workflow requires a hosted Temper service, and cached projects remain
usable without network access.

## Setup and checks

Requirements: Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --dev --locked
python scripts/temper-gate.py all
```

If `uv` is not installed and a temporary download is acceptable, the gate can
bootstrap it for that invocation only:

```powershell
python scripts/temper-gate.py --bootstrap-uv temp all
```

The individual checks are available when iterating on a focused change:

```powershell
python scripts/temper-gate.py format
python scripts/temper-gate.py lint
python scripts/temper-gate.py typecheck
python scripts/temper-gate.py compile
python scripts/temper-gate.py unit
python scripts/temper-gate.py maintenance
python scripts/temper-gate.py diff
```

`python scripts/temper-gate.py fixture-help` describes the deterministic
fixture walkthrough entry point.

## Library-backed local runtime

Slice 8 adds a Temper-owned LoRA runtime port with three implementations:

- `DeterministicLibraryBackend` exercises the complete service, artifact,
  evaluation-inference, and local-use contract without ML libraries or
  hardware.
- `WslWorkerBackend` is the reference Windows-hosted port. It invokes one
  explicitly named WSL distribution without a shell and uses an explicitly
  mapped shared staging root for immutable requests, typed lifecycle messages,
  cooperative control markers, and content-verified transfers.
- `TransformersPeftBackend` is the local PyTorch, Transformers, PEFT, and
  Accelerate implementation. Running it directly is the secondary native
  execution path; it is eligible only when its probe proves the selected
  execution target and recipe requirements.

Install the optional library stack in the environment that runs the real
backend. The default development and fixture gate does not install it:

```powershell
uv sync --dev --extra runtime --locked
```

Model and tokenizer inputs must already be local directories. The runtime uses
local-only library loading with remote code disabled; it does not download a
model or silently switch between WSL and native execution. The caller supplies
one explicit target class, local model/tokenizer sources, and, for WSL, one
Windows/WSL staging-root mapping and worker Python executable. Capability facts
are converted into the existing hardware profile and must pass the existing
preflight checks before a request is frozen.

The worker never opens the canonical project store. Windows-side application
services alone ingest verified messages, transfer receipts, checkpoints, and
the normalized three-member adapter bundle. A partial transfer, conflicting
message replay, missing terminal response, or worker loss cannot complete a
run or admit an artifact. The detailed ownership and failure model is recorded
in `docs/superpowers/plans/2026-07-18-slice-8-runtime-architecture.md`.

The opt-in ROCm smoke test documents the required private environment variable
names without embedding machine values:

```powershell
$env:TEMPER_RUN_SLICE8_HARDWARE = "1"
uv run --extra runtime pytest -m hardware tests/hardware/test_slice8_real_adapter.py
```

It skips truthfully unless every explicitly configured local prerequisite is
present. No hardware test is required by the normal repository gate.

## Local evaluation lab

Slice 7 adds a dependency-free loopback UI for the complete deterministic
fixture journey:

```powershell
uv run temper ui .temper-fixture-output
```

Open the printed `127.0.0.1` URL, then move through project setup, dataset
inspection, recipe resolution and preflight, two fixture runs, synchronized
comparison and review, evidence-backed recommendation, explicit selection,
focused or batch local use, case capture, and verified adapter export. The UI
binds only to numeric loopback addresses and all canonical writes go through
the existing application services. Project state remains inspectable with the
CLI commands below.

This lab is offline fixture behavior. It does not load a real model, create a
deployment, provide chat memory, or require an external dashboard. Those
boundaries are surfaced in the UI instead of being implied capabilities.

## Evidence inspection

The committed minimal project exercises the canonical store without private
data, hardware, network access, or external services:

```powershell
uv run temper status fixtures/projects/minimal
uv run temper verify fixtures/projects/minimal
uv run temper dump fixtures/projects/minimal
uv run temper manifest fixtures/projects/minimal --type project --id project-minimal
```

`status` and `verify` are read-only and prove record identities, reference
closure, event chains, bundle-manifest identities, and derived-state
rebuildability. `dump` emits the deterministic, identity-free projection in
`schemas/public/public-dump-v1.schema.json`; because v1 records carry no public
classification, their payload fields are omitted by default. `manifest` emits
the exact selected canonical record for local inspection; its output is not a
public export.

Successful commands write one canonical JSON document to standard output.
Failures write only a stable JSON error code to standard error: exit `1` for an
invalid or unverifiable subject, `2` for usage, `3` for a public-safety refusal,
and `4` for a filesystem or internal failure. `uv run temper --version` remains
available for version inspection.

## Fixtures and hardware tests

Tests, fixtures, and examples must be small, synthetic, deterministic, and
safe to publish. A fixture may use a hash only when the exact synthetic source
bytes are committed and reviewable. Do not add model weights, checkpoints,
runtime stores, generated review packets, real datasets, or production logs.

Hardware-dependent tests are opt-in. They must not be required by the default
gate and must skip with an explicit reason when the needed local capability is
unavailable. The normal fixture path must run without a GPU, network access, or
external service.

## Public-repository safety

This is a public repository. Never commit credentials, private URLs, local
paths, usernames, host or device identifiers, private artifacts, or records
derived from them. Use synthetic identities and inert placeholders in all
examples. Review `AGENTS.md` before staging, committing, or pushing changes;
it contains the complete public-safety rules.
