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
