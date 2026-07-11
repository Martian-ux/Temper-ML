# Temper ML v1 Execution Roadmap

**Status:** Adopted governing execution roadmap
**Date:** 2026-07-09
**Last aligned:** 2026-07-11
**Product architecture:**
- docs/superpowers/specs/2026-06-30-temper-ml-architecture-design.md
**Historical planning reference:**
- docs/superpowers/plans/2026-07-01-temper-ml-v0.1-implementation-plan.md

This roadmap is the authoritative v1 execution sequence. The architecture
document remains the approved source for product boundary and design decisions.
The July 1 plan remains available for stable public links and history, but its
earlier workstream order is superseded and must not be used to schedule work.

## 1. Planning Baseline

The current repository implements only the Python package skeleton, canonical JSON,
content projections, a local write-once store, a minimal CLI, and their unit
tests. No project service, dataset pipeline, adapter runtime, UI, evaluation
system, retention feature, loop, merge workflow, or compatibility backend
exists yet.

This is useful: v1 can be built from the inside out without migrating a
provider-led product. The canonical store remains the foundation, but new
features must not bypass it.

## 2. Delivery Rules

- Every slice owns one visible or testable behavior and leaves the main gate
  green.
- Domain records and services precede UI routes and runtime integrations.
- Fixture behavior proves a contract before hardware-dependent code is added.
- The native Temper runtime remains the reference implementation. External
  engines are compatibility adapters only.
- Core training, evaluation, local adapter use, replay, and evidence workflows
  require no hosted Temper service.
- The first real-hardware target is a Windows host with an explicit WSL2 Ubuntu
  ROCm worker. Native Windows execution is secondary and capability-gated
  behind the same runtime contract.
- Public tests use small synthetic fixtures. Hardware tests are opt-in and must
  skip with an explicit reason when unavailable.
- Every artifact, evaluation, cleanup, replay, loop, merge, readiness
  assessment, and user override emits immutable evidence.
- A feature is incomplete until its failure modes and recovery behavior have
  tests.

## 3. Critical Path

    canonical store
      -> project, task, compatibility, recipe, and experiment contracts
        -> deterministic dataset pipeline
          -> fixture adapter runtime and artifact verification
            -> evaluation, recommendations, and review
              -> early UI, CLI, and local-use vertical slice
                -> library-backed local adapter runtime
                  -> retention and replay
                    -> loops, merges, and readiness
                      -> optional Noah and external compatibility

The early usable milestone is the fixture vertical slice. The first
real-hardware usable milestone trains, evaluates, selects, and uses an adapter
through the Windows/WSL2 ROCm topology. The full v1 milestone is the local
library-backed runtime plus the evidence, retention, iteration, merging, and
readiness capabilities around it.

Slices 0 through 10 are all required v1 delivery. Their milestones provide
usable checkpoints; they do not defer accepted scope to a later product
version. Slice 11 is an optional compatibility boundary and is not required to
declare v1 complete.

## 4. Implementation Slices

### Slice 0: Repository and Gate Baseline

**Goal:** Keep development reproducible while the product surface expands.

**Implement:**
- Complete README setup and product-boundary guidance.
- Add formatting, linting, typing, and test commands to the existing gate.
- Add CI that runs the cross-platform gate on pushed changes.
- Define fixture naming, public-safety review, and hardware-test conventions.

**Proof:**
- A clean checkout runs setup, static checks, unit tests, and fixture help.
- CI does not require private datasets, hardware, or external services.

**Depends on:** nothing.

### Slice 1: Complete Canonical Store and Evidence Services

**Goal:** Finish the storage primitives before creating product records.

**Implement:**
- Canonical store layout, typed record envelopes, projection registry, and
  byte and bundle verification.
- Append-only event stream with predecessor hashes, idempotency keys, and
  derived-state rebuilding.
- Redaction service and public-safe dump/export behavior.
- CLI commands for status, dump, verify, and manifest inspection.
- Recovery, corruption, symlink, duplicate, and interrupted-write tests.

**Proof:**
- A synthetic project can be verified, dumped, and reconstructed after a
  simulated interrupted write.
- Canonical records survive derived-state rebuilding and rejected cleanup
  attempts.

**Depends on:** Slice 0.

### Slice 2: Core Domain Contracts

**Goal:** Define v1 concepts before implementing behavior around them.

**Implement:**
- Typed schemas and projection versions for Project, ProjectPolicy,
  TaskDefinition, BaseModelRevision, CompatibilityGroup, Recipe,
  RecipeResolution, HardwareRequirements, HardwareCapabilityProfile,
  ExecutionTarget, Experiment, Run, Artifact, ArtifactAvailability,
  LocalUseSession, and AdapterExport.
- Compatibility relationship validators for comparison, merge, resume, and
  deployment targets.
- Baseline policy records for per-model, project-champion, and fixed-reference
  comparisons.
- Immutable derivation records and manifest-diff representation.

**Proof:**
- Contract tests reject incompatible model, tokenizer, target-module, and
  runtime-target combinations.
- A derived experiment is distinct from its parent and exposes an exact reason
  and manifest diff.

**Depends on:** Slice 1.

### Slice 3: Project, Recipe, Hardware, and Experiment Services

**Goal:** Make a task-centered project and a resolved experiment real.

**Implement:**
- Project creation and opening services.
- Recipe catalog, explicit expert overrides, and deterministic recipe
  resolution.
- Hardware capability capture, constraint resolution, preflight estimates, and
  material-change detection.
- Windows/WSL execution-target selection, portable path mapping, and visible
  platform-change derivation rules.
- Experiment freeze, clone, strict replay plan, and assisted adapted-replay
  plan services.
- CLI views for project status, recipe resolution, preflight, and manifest
  diff.

**Proof:**
- The same recipe and inputs resolve to the same manifest.
- A changed machine either accepts strict replay unchanged or creates a labeled
  derived experiment; it never silently rewrites the original.

**Depends on:** Slice 2.

### Slice 4: Deterministic Dataset Pipeline

**Goal:** Produce immutable LLM-training dataset versions.

**Implement:**
- JSON, JSONL, CSV, and supported Hugging Face import adapters.
- Field mapping, record validation, exclusion receipts, deterministic
  filtering, and deduplication.
- Token-length analysis, deterministic split assignment, exact rendered
  training text, previews, and summary statistics.
- Correction-report and reimport-comparison services.
- Dataset version, renderer, tokenizer, and split identity projections.

**Proof:**
- Synthetic fixtures cover invalid rows, duplicate rows, long examples,
  different field mappings, and reimported source data.
- Re-running the same import produces byte-identical evidence and split
  membership.

**Depends on:** Slices 1 and 2.

### Slice 5: Temper-Owned Fixture Adapter Runtime

**Goal:** Prove the normal runtime contract without a GPU or external trainer.

**Implement:**
- Runtime preflight, resolved-request, launch, progress, checkpoint,
  cancellation, interruption, recovery, artifact-ingestion, and terminal-event
  services.
- Deterministic fixture adapter trainer that consumes the actual resolved
  experiment manifest.
- Artifact integrity verifier for bytes, structure, base-model compatibility,
  tokenizer compatibility, and provenance.
- Fixture inference runtime, focused local-use sessions, local batch inference,
  verified export, and run-log capture.
- End-to-end CLI workflow from project creation through verified artifact.

**Proof:**
- A fixture project completes through native services with no network, GPU,
  external dashboard, or external trainer.
- A verified fixture adapter can be used interactively or in a local batch and
  exported without becoming a hosted deployment.
- Cancellation and recovery leave coherent append-only evidence.
- Integrity validation still runs when the selected evaluation mode has no
  quality evaluation.

**Depends on:** Slices 3 and 4.

### Slice 6: Evaluation, Recommendation, and Review Services

**Goal:** Make selection evidence honest and policy-driven.

**Implement:**
- Evaluation modes, deterministic evaluators, held-out loss, task metrics,
  format checks, and baseline comparison records.
- Development, regression, and confirmation suites with soft-seal state
  transitions and contamination disclosure.
- Policy engine for hard qualifiers, advisory metrics, objectives,
  lexicographic ranking, Pareto alternatives, and confidence labels.
- Separate evidence-status and user-decision events, including optional
  override reasons.
- Structured solo review records and optional blind-review packet, leak-audit,
  and sealed-reveal services.
- Validation that rejects model-judge evaluators from v1 policies.

**Proof:**
- A candidate can pass, fail, be inconclusive, or be subjective-only without
  losing the underlying evidence.
- Inspecting or changing a confirmation case changes the reported evidence
  state.
- A blind review remains optional; structured solo review can produce valid
  recorded evidence.

**Depends on:** Slice 5.

### Slice 7: Evaluation Playground and Early UI

**Goal:** Deliver the first coherent local product experience.

**Implement:**
- Loopback server and UI routes backed only by application services.
- Project setup, dataset import, recipe selection, resolution preview,
  hardware preflight, run status, logs, metrics, and artifact inspection.
- Playground with side-by-side comparison, synchronized prompts, optional
  hidden identities, inference controls, saved outputs, notes, ratings, prompt
  replay, and conversion of discovered failures into cases.
- Focused local-use view for one selected adapter, local batch execution,
  explicit capture into evaluation cases, and verified adapter export.
- Recommendation, evidence, user-decision, and registry views.
- UI boundary tests that prove routes cannot write canonical records directly.

**Proof:**
- A user completes the fixture vertical slice entirely through Temper's UI.
- The user can move from a selection decision to focused local use without
  creating a deployment or general chat surface.
- The same project state remains inspectable through the CLI.
- The UI is not a general chat surface and exposes no external dashboard as a
  required workflow.

**Depends on:** Slice 6.

### Slice 8: Library-Backed Local Adapter Runtime

**Goal:** Replace fixture training internals with the supported local runtime.

**Implement:**
- Library integration for tokenizer loading, PEFT adapter creation, training,
  acceleration, quantization, checkpointing, and metrics.
- A Windows-hosted runtime port and WSL2 Ubuntu ROCm worker for hardware probes,
  training, evaluation inference, local-use inference, and cancellation.
- Immutable request handoff, local staging, transfer verification, ingestion
  receipts, heartbeat, reconnection, and recovery across the Windows/WSL
  boundary. The WSL worker never writes canonical project records directly.
- Preflight estimators and recipe constraint checks for supported local
  hardware.
- Library-version, model, tokenizer, device, checkpoint, artifact, log,
  cancellation, recovery, and failure evidence.
- Deterministic local doubles for contract tests and capability-gated real
  hardware tests.
- A secondary native Windows PyTorch/ROCm execution path only for combinations
  that pass explicit capability checks and the same runtime contract.
- Artifact ingestion that preserves the same manifest and lifecycle contracts
  used by the fixture runtime.

**Proof:**
- A capability-gated real adapter run is represented by the same project,
  experiment, run, artifact, and evaluation records as a fixture run.
- A supported adapter is trained, evaluated, selected, and used locally through
  the Windows/WSL2 ROCm reference topology.
- Interrupted staging or worker loss cannot produce a successful run or an
  unverified artifact, and reconnection does not duplicate the run.
- No external trainer-specific manifest or identifier becomes canonical.

**Depends on:** Slices 5 through 7.

### Slice 9: Retention, Cleanup, and Reproduction

**Goal:** Make local storage and reproducibility truthful.

**Implement:**
- Full-retention default, byte inventory, shared-reference accounting, and
  artifact-availability records.
- Cleanup planner and executor with impact warnings for resumability,
  inspectability, final-artifact availability, cache convenience, and debugging
  evidence.
- Immutable cleanup receipts and lifecycle events.
- Strict replay execution and adapted-replay derivation flows.
- UI and CLI views for inventory, cleanup preview, cleanup result, replay
  planning, and manifest differences.

**Proof:**
- Cleanup frees only selected heavy bytes and preserves canonical evidence.
- A replay is either strict and identical or visibly adapted and derived.
- A removed checkpoint cannot be advertised as resume-available.

**Depends on:** Slice 8.

### Slice 10: Bounded Loops, LoRA Merging, and Readiness

**Goal:** Add disciplined iteration and handoff assessment.

**Implement:**
- Manual next-trial workflow and optimized loop policy with search space, hard
  qualifiers, objectives, budgets, stopping conditions, and resource limits.
- Trial queue, cancellation, accounting, and confirmation-suite evaluation of
  the selected candidate.
- Pareto views for qualified tradeoffs.
- LoRA merge planner and executor with compatibility validation, maintained
  method registry, parent integrity verification, immutable lineage, and
  parent-comparison evaluation.
- Runtime-target readiness policies, required checks, and ready,
  approved-by-user, not-ready, and unevaluated assessments.

**Proof:**
- No automated loop can start without explicit bounds.
- A merged candidate cannot be recommended until it is compared with each
  parent.
- Readiness assessment never controls a deployment runtime.

**Depends on:** Slice 9.

### Slice 11: Optional Compatibility Boundaries

**Goal:** Add Noah or other engines only when a measured v1 product need
justifies them, without weakening the primary product.

**Implement:**
- Versioned compatibility schemas, synthetic fixtures, request translation,
  external-reference ingestion, and public-safe redaction.
- Noah-specific dataset, policy, and evaluation compatibility package.
- Optional external-trainer bridge only where it preserves Temper's manifest,
  lifecycle, evidence, and ownership rules.
- Compatibility UI that clearly identifies external evidence without making the
  user leave Temper.

**Proof:**
- Normal v1 projects remain fully usable with every compatibility integration
  disabled.
- Contract tests prove external IDs and files cannot replace Temper records.

**Depends on:** Slice 10.

This slice is compatibility scope, not a prerequisite for v1 acceptance. It may
ship in v1 only when a concrete capability gap is demonstrated and every core
workflow remains complete without it.

## 5. Cross-Cutting Test Strategy

Every slice adds all applicable test layers:

| Layer | Purpose |
| --- | --- |
| Unit | Pure domain rules, canonical projections, state transitions, and policy decisions |
| Contract | Schema, port, runtime, and compatibility boundaries |
| Integration fixture | Full deterministic workflows without hardware or network |
| Adversarial | Corruption, path safety, redaction, lifecycle, cleanup, and replay failures |
| UI | Service boundaries, user-visible states, empty/error paths, and accessibility |
| Hardware | Opt-in Windows/WSL2 ROCm and native-Windows contract execution with explicit capability skips |

The primary gate remains:

    python scripts/temper-gate.py all

When the local machine lacks uv, the documented temporary bootstrap is:

    python scripts/temper-gate.py --bootstrap-uv temp all

## 6. First Implementation Batch

Start with the smallest change set that unlocks every later module:

1. Finish Slice 1 storage services: layout, event stream, verifier, redaction,
   and inspect CLI commands.
2. Add Slice 2 schemas and contract tests for project, task, compatibility,
   recipe, hardware, execution target, experiment, run, artifact, local-use
   session, and export records.
3. Implement Slice 3 project and recipe-resolution services only after those
   schemas are stable.
4. Define the Windows-host/WSL-worker port and its failure semantics before
   adding ROCm-specific code.
5. Do not add UI, PyTorch dependencies, a provider adapter, or hardware code
   before the fixture contracts exist.

The first end-to-end milestone is Slice 5. It is the earliest point at which
Temper can honestly demonstrate its central promise with deterministic evidence.

## 7. Completion Checkpoints

| Checkpoint | Completion evidence |
| --- | --- |
| Foundation complete | Canonical storage, verification, redaction, recovery, and CLI inspection |
| Fixture product complete | Project through verified artifact, evaluation, recommendation, review, and UI |
| Local-use product complete | A selected verified adapter supports focused interactive use, local batch inference, and verified export |
| Local runtime complete | Library-backed training, evaluation, and use through the Windows/WSL2 ROCm topology using the same manifest and evidence contracts |
| Reproducibility complete | Retention, cleanup, strict replay, and adapted replay are explicit and tested |
| Iteration complete | Bounded loops, merge lineage, confirmation evaluation, and readiness assessment |
| v1 complete | All architecture acceptance criteria pass without an external trainer or dashboard |

## 8. Deferred Work

Do not begin these until v1 proves a product need:

- full-model training or pretraining;
- general ML workloads;
- model-judge evaluation;
- distributed orchestration;
- a hosted control plane, teams, permissions, or billing;
- hosted inference endpoints or remote project execution;
- a general chat client;
- a generalized plugin marketplace; and
- external trainer replacement of the native Temper runtime.
