# Temper ML

Temper ML is a local-first product for LLM adapter experimentation. It is being
built to help one semi-technical user create, evaluate, compare, reproduce, and
retain adapter experiments without making an external trainer, dashboard, or
cloud account the source of truth.

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
fixture walkthrough entry point. The public CLI currently exposes its version
with `uv run temper --version`.

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
