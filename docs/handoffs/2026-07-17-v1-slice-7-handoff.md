# Temper ML v1 Slice 7 handoff

**Status:** Slice 7 implementation candidate; publication and integration
remain maintainer-gated.

**Base:** `c5b752600473f49d5411eea39d2c2301a4cdd61c`

**Branch:** `codex/slice-7-evaluation-playground`

This handoff covers only Slice 7 of the adopted v1 roadmap. It does not start
the Slice 8 library-backed runtime, change a published contract, or authorize
merge, release, deployment, or production mutation.

## Delivered product journey

- A dependency-free HTTP server binds only to numeric loopback addresses and
  serves packaged HTML, CSS, JavaScript, and a same-origin JSON API.
- A dense responsive dashboard summarizes verified store state, workflow
  progress, candidate manifests, runtime and integrity evidence, the current
  registry decision, and the next bounded action. Desktop, tablet, and narrow
  layouts preserve the same journey without horizontal page scrolling.
- The UI walks through fixture project setup, local dataset import and private
  preview, two distinct recipe resolutions, hardware preflight, verified run
  status, public-safe events, metrics, and artifact evidence.
- The playground runs one synchronized prompt and inference settings against
  both candidates. Structured solo review is sufficient; the optional blind
  path prepares a leak-audited packet, seals the judgment, then reveals the
  candidate mapping.
- Evaluation exposes development and confirmation evidence, disclosed
  conflicts, and a deliberately low-confidence recommendation when the
  synthetic candidates tie. A separate user decision controls registry state.
- The selected adapter can be used for focused local inference or a local
  batch, a specifically identified completed review can be captured as an
  evaluation case, and an adapter export is verified without asserting
  deployment readiness.
- The same canonical project remains inspectable through the existing CLI.
  Restart views distinguish persisted evidence from prepared dataset bytes
  that existed only in the prior process.

## Boundaries and service ownership

The HTTP package has no canonical-store import. GET routes are derived
read-only views, while every mutation calls the Slice 7 journey application
service, which composes the accepted project, dataset, recipe, experiment,
run, evaluation, local-use, and export services. Raw prompts and dataset
previews are not copied into public projections.

The UI remains a thin local fixture surface. It is not general chat, a hosted
service, a deployment controller, an external dashboard, or a real-model
runtime. No dependency or provider was added.

## Changed paths

- `src/temper_ml/app_services/fixture_journey.py` owns staged fixture
  orchestration and verified aggregate reads.
- `src/temper_ml/app_services/local_use.py` records optional evaluation-case
  capture provenance on saved local-use sessions and exposes the same
  byte-level artifact verification used before local inference.
- `src/temper_ml/ui/` owns the hardened loopback transport and packaged UI.
- `src/temper_ml/cli.py` adds `temper ui` and delegates the existing fixture
  workflow to the shared journey service.
- `tests/unit/test_fixture_journey.py`, `tests/unit/test_ui_server.py`, and
  `tests/integration_fixture/test_slice7_ui_vertical_slice.py` cover service
  staging, artifact loss or corruption, explicit review identity capture,
  transport boundaries, the browser-facing API journey, CLI coexistence,
  restart behavior, and public projection safety.
- `README.md` documents the local lab and its non-capabilities.

## Verification contract

Iteration uses focused service, HTTP-boundary, Slice 5 compatibility,
local-use contract, and complete Slice 7 vertical-slice tests. Formatting
precedes the final verification pass. The publication candidate must pass one
full repository gate on its exact immutable bytes:

```text
python scripts/temper-gate.py --bootstrap-uv temp all
```

The draft pull request must also pass both required Ubuntu and Windows jobs.
Reviewers and integrators may reuse only evidence bound to the exact candidate
identity and command semantics.

## Integration guidance

The candidate is one root-writer change with a single independent cold review
at the UI, CLI, and application-service boundary. The review and full-gate
results support a maintainer integration decision but cannot grant it. Do not
merge, delete recovery material, or begin Slice 8 without explicit maintainer
authorization.
