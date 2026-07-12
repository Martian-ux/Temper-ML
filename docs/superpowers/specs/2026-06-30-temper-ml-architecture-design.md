# Temper ML v1 Architecture Decisions

**Status:** Approved product direction, superseding the former provider-first architecture
**Date:** 2026-07-09
**Last aligned:** 2026-07-11
**Authority:** Product-grill decisions and subsequent maintainer clarifications are the governing v1 specification.
**Scope:** Local-only LLM adapter experimentation for one semi-technical user.

This document governs v1 product boundary and design decisions. The adopted
implementation order is the July 9 execution roadmap at
`docs/superpowers/plans/2026-07-09-temper-ml-v1-execution-roadmap.md`; the
July 1 plan is retained only as a historical planning reference.

## 1. Product Boundary

Temper v1 is an LLM adapter experimentation product. It helps a user train,
evaluate, compare, reproduce, retain, select, and use adapters locally without
managing scripts, trainer dashboards, artifact folders, and informal
evaluation notes.

The primary workflows are:

1. Create and train an adapter.
2. Evaluate, select, and use an adapter locally.
3. Iterate reproducibly through cloning, replay, bounded experiment loops, and
   compatible adapter merging.

Using an adapter locally means loading an integrity-verified adapter with its
compatible base model and tokenizer for focused interactive or batch inference.
Temper preserves the selected artifact, inference settings, prompt or batch
inputs when saved, outputs when saved, and provenance. It can export a verified
adapter bundle and manifest for another compatible local runtime. This workflow
does not create a hosted endpoint, manage a production runtime, add assistant
memory, or turn the playground into a general chat client.

Temper v1 is not a general machine-learning platform. It does not support
classical ML, arbitrary PyTorch programs, full-model pretraining, a general
deployment system, a general chat client, or a hosted control plane. v1 must
support LoRA adapters at minimum; the merge workflow is LoRA-only.

## 2. Governing Principles

- Temper owns the normal user workflow, project identity, dataset identity,
  manifests, run lifecycle, metrics, artifacts, retention state, and evidence.
- PyTorch, Transformers, PEFT, Accelerate, quantization libraries, optimizers,
  and tokenizers provide ML machinery. They do not own Temper's product model.
- An external trainer, tracker, or runtime is never Temper's source of truth.
  Its IDs, files, logs, and metrics are provider evidence or external
  references attached to Temper records.
- Normal users never need an external dashboard to complete a supported
  workflow. Optional integrations remain subordinate to Temper's UI and
  records.
- Core workflows are local-only. They require no Temper cloud account, hosted
  control plane, remote telemetry service, or network service after explicitly
  imported models, datasets, and dependencies are available locally.
- Canonical records are immutable and hash-addressed. Derived views, caches,
  and local indexes are rebuildable conveniences.
- Evidence and user decisions are separate. An override cannot rewrite or hide
  the evidence that preceded it.
- Every automated action is bounded and explainable. No v1 operation silently
  changes a scientific intention, a project policy, or a production system.

The dependency direction is:

    Temper UI
      -> Temper application services
        -> Temper domain and canonical store
          -> Temper-owned adapter runtime
            -> PyTorch / Transformers / PEFT / Accelerate / tokenizer and quantization libraries
          -> isolated compatibility backends and optional integrations

## 3. Runtime Ownership and Compatibility Boundaries

### Temper-Owned Runtime

Temper owns the primary v1 training runtime behavior:

- resolution of recipes into manifests;
- preparation of the rendered training dataset;
- launch, cancellation, recovery, checkpoints, retention, and ingestion;
- lifecycle events, metrics, logs, artifact verification, and provenance; and
- the UI and CLI behavior surrounding a run.

The runtime may call mature libraries for training, tokenization, scheduling,
optimization, checkpoint serialization, and quantization. It captures library
versions and resolved runtime configuration in run evidence.

### Reference Windows/WSL ROCm Topology

The first supported real-hardware topology is a Windows host with a WSL2 Ubuntu
worker using ROCm on a supported AMD GPU. Temper's local launcher, loopback UI
and CLI, application services, and canonical project store remain authoritative
on the Windows side. Training, evaluation inference, local-use inference, and
hardware probes execute through an explicit WSL worker boundary.

That boundary follows these rules:

- Temper freezes an immutable runtime request before launch. The worker cannot
  silently alter it or write canonical project records directly.
- Inputs and outputs cross through explicit staging and ingestion operations.
  Artifact identities, byte counts, and transfer receipts make interrupted or
  partial transfers detectable.
- Lifecycle messages cover launch, progress, heartbeat, cancellation,
  interruption, reconnection, recovery, completion, and failure.
- Canonical manifests use portable identities and logical locations rather
  than Windows drive paths, WSL distribution paths, or host-specific names.
- Hardware and platform capability belong to the run. A platform-driven
  material configuration change creates a visibly derived experiment.
- The UI exposes the selected execution target, detected capabilities,
  resolved changes, estimates, and unsupported combinations before launch.

A native Windows PyTorch/ROCm worker may use the same runtime port when
capability detection proves that the required library and GPU combination is
supported. It is a secondary execution target, not a second product model, and
Temper must never silently switch between native Windows and WSL execution.

The process and transfer protocols are internal ports, not a distributed
orchestration feature. Both sides remain on one local machine, bind management
interfaces to loopback, and require no hosted Temper service.

### Compatibility Backends

External trainer products may be studied or used as isolated compatibility
backends when a measured capability gap justifies them. They are not the normal
v1 path, cannot define the primary manifest schema, and cannot become
authoritative for artifacts, metrics, lifecycle state, or recommendations.

Noah may remain the first compatibility workflow and fixture source, but its
fields, thresholds, dataset behavior, and existing trainer stay outside Temper
core. Noah compatibility follows the generic adapter-runtime vertical slice; it
does not gate it.

## 4. Project, Task, Models, and Compatibility

### Project and Task Contract

A Temper project represents one adapter purpose or task, not a general lab
folder. Its immutable project policy binds:

- task definition and rendering contract;
- evaluation policy and case suites;
- readiness policy;
- retention policy;
- approved recipe families; and
- baseline and recommendation policy.

A project may contain multiple base models. All artifacts in the project remain
interpretable against the same primary task definition, evaluation policy,
regression cases, confirmation suites, and readiness expectations.

### Compatibility Groups

Each base-model and adapter configuration belongs to explicit compatibility
groups. A group records at least the base-model revision, tokenizer revision,
chat or rendering template, adapter type, target modules, and applicable
runtime-target constraints.

Temper records relationships separately so it never infers compatibility from a
friendly model name:

- **comparable:** candidates can be evaluated against the same declared task
  policy and compared through its objectives;
- **merge-compatible:** adapters satisfy the required base-model, tokenizer,
  adapter-type, target-module, and merge-method constraints;
- **resume-compatible:** a run can safely use a retained checkpoint and its
  exact training state; and
- **deployment-compatible:** an artifact has passed checks for a declared
  runtime target.

### Baselines

Temper supports three independent baseline concepts:

- **per-model baseline:** whether an adapter improves its own base model;
- **project champion:** whether a candidate improves the current best artifact
  for the task under the comparison policy; and
- **fixed reference baseline:** whether a candidate improves a stable,
  policy-pinned project reference.

Each evaluation result states which comparisons were applicable, executed,
unavailable, or invalid because the compatibility policy did not permit them.

## 5. Dataset Pipeline

Temper owns a narrow, deterministic pipeline for LLM adapter training. v1
imports JSON, JSONL, CSV, and supported Hugging Face datasets through explicit
import adapters. It does not become a spreadsheet editor or a general dataflow
engine.

For each dataset version, Temper records:

1. source descriptors and import contract;
2. field mapping and task rendering configuration;
3. validation results, invalid-row exclusions, and reasons;
4. deterministic filtering and deduplication rules;
5. token-length analysis and exact tokenizer identity;
6. deterministic split rules and resulting membership;
7. the exact rendered training text or a reproducible rendering projection;
8. preview examples and summary statistics; and
9. immutable content identity and provenance.

Temper may diagnose data quality and issue correction reports. Users correct an
authoritative source outside Temper, then reimport and compare versions.
Unsupported transformations happen outside Temper through a strict
import-and-provenance contract.

## 6. Recipes, Hardware, Experiments, and Runs

### Recipe-First Configuration

Temper presents versioned recipes rather than an empty expert form. A recipe
captures user-facing choices such as training profile, adapter size, memory
mode, quantization, training duration, checkpoint policy, evaluation intensity,
and retention policy.

Recipe resolution produces an immutable manifest containing exact technical
settings, including adapter targets, rank, alpha, dropout, learning rate,
effective batch size, sequence length, optimizer, precision, gradient
accumulation, seed, schedule, checkpoint cadence, and relevant library
versions. Expert overrides are explicit recipe inputs, never hidden mutations.

### Hardware-Aware Resolution

Recipes resolve against declared machine constraints and a captured hardware
capability profile. Before launch, Temper shows every resolved change, estimate,
and constraint that affected it.

Hardware identity belongs to a run. Hardware requirements and resolved training
configuration belong to an experiment. Replaying on a different machine must
not silently change an experiment. If the original manifest cannot run
unchanged, Temper offers an assisted adaptation that creates a derived
experiment with a visible manifest diff and an adapted-reproduction label.

Resolution also binds an execution-target class such as WSL2 ROCm or native
Windows ROCm. Moving between target classes is never treated as an invisible
retry, even when both targets report the same GPU architecture.

### Experiment and Run Semantics

An experiment is immutable scientific intention: the task and project-policy
revisions, dataset version, base-model and tokenizer identities, recipe and
resolution, evaluation policy, compatibility group, and declared hardware
requirements.

A run is one execution attempt of an experiment. Retries after interruption
create new runs under the same experiment. Changes to a seed, dataset version,
adapter setting, training steps, runtime or library version, recipe resolution,
or hardware-driven training configuration create a derived experiment.

Temper supports two reproduction modes:

- **strict replay:** run the original resolved manifest unchanged; and
- **assisted adapted replay:** derive and label a new experiment when the
  original cannot run on the available machine.

Adapted results must never be presented as exact reproductions.

## 7. Artifacts, Retention, and Cleanup

An adapter artifact records its adapter type, content identity, base-model and
tokenizer identities, compatibility groups, producing run, parent artifacts,
storage locations, integrity evidence, and lineage.

Full retention is the default. Canonical manifests, hashes, lifecycle events,
final metrics, recommendation evidence, and cleanup records are retained by
ordinary cleanup. Heavy bytes may include checkpoints, optimizer state, cached
datasets, duplicated model files, rendered-data caches, logs, and final adapter
files.

Temper provides a cleanup menu that calculates, before deletion:

- physical bytes freed after shared references;
- retained and deleted byte classes;
- loss of resumability, inspectability, final-adapter availability, cached
  dataset convenience, or debugging evidence; and
- affected runs and artifacts.

Deleting heavy bytes creates immutable lifecycle events and cleanup receipts. It
never changes a canonical manifest or falsely implies that an artifact is still
locally available.

## 8. Evaluation, Review, and Playground

### Evaluation Modes and Integrity

Before a run, a user selects one of: no quality evaluation, light evaluation,
full suite, or an experiment loop. Every mode performs artifact-integrity
validation after training: verify the artifact exists, hashes correctly, loads
against the intended base model, has the expected structure, and has complete
provenance. Quality evaluation is separate from integrity validation.

v1 uses deterministic checks, held-out loss, task metrics, format checks,
development cases, regression cases, confirmation cases, and recorded human
review. v1 does not use model judges or an unexplained universal quality score.

### Case Suites and Soft Sealing

- **development cases** guide iteration;
- **regression cases** prevent known failures from returning; and
- **confirmation cases** support final validation after candidate selection.

Confirmation suites are soft-sealed, not access-controlled secrets. Inspecting
or modifying them changes their evidence state. States include sealed,
unsealed, modified, contaminated, and retired; reports must disclose the state
used by every recommendation.

### Human Review and the Playground

Structured solo review in the playground is valid v1 recommendation evidence
when its prompts, settings, outputs, notes, ratings, and reviewer declarations
are recorded. Formal blind packets are stronger evidence when appropriate but
are optional. When blind review is used, Temper performs a mechanical leak audit
and requires sealed judgments before reveal.

The playground is an evaluation instrument, not a general-purpose chat client.
It supports side-by-side comparison, synchronized prompts, optional hidden
identities, inference controls, saved outputs, notes, ratings, prompt replay,
and conversion of discovered failures into development or regression cases. It
does not provide assistant memory, character chat, or a daily chat experience.

### Local Adapter Use

After artifact integrity succeeds, a user can open a selected adapter in a
focused local-use session or run a local batch. Temper verifies the base model,
tokenizer, adapter structure, and runtime target again before loading. Saved
sessions retain the exact artifact, inference settings, inputs, outputs, and
runtime evidence; ephemeral use need not become recommendation evidence.

Local-use output becomes evaluation or regression evidence only through an
explicit capture action. Export produces the adapter bytes, an integrity
manifest, compatibility requirements, and provenance needed by a compatible
local runtime. Export does not imply deployment readiness.

## 9. Recommendations, Registry, and Readiness

Recommendations are policy-based. A policy declares hard qualifiers, advisory
metrics, optimization objectives, baseline comparisons, and readiness checks.
Temper reports conflicting results explicitly rather than collapsing them into a
hidden quality score.

Evidence status and user decision status are distinct:

| Evidence status | User decision status |
| --- | --- |
| passed policy | selected |
| failed policy | rejected |
| inconclusive | pinned |
| unevaluated | deprecated |
| contaminated | archived |
| subjective-only | deployment override |

Users may select or approve a warned artifact, but the override does not alter
the evidence. Temper strongly prompts for an optional override reason and
records it as a separate decision event.

Deployment readiness is an assessment, not a deployment system. For a declared
runtime target, Temper reports:

- **ready:** required checks passed;
- **approved by user:** user approved despite evidence warnings;
- **not ready:** required checks failed; or
- **unevaluated:** no readiness decision is possible.

Temper v1 does not manage production runtime state, rollout, traffic, or
production mutation.

## 10. Iteration, Optimization, and Merging

Manual iteration lets a user choose the next experiment. Optimized iteration
uses a declared search space, hard qualifiers, objectives, budgets, and
stopping conditions to select the next trial.

Every automated loop has explicit limits on one or more of trial count, wall
time, GPU time, disk budget, and stopping criteria. The loop emits normal
experiments and runs; it cannot hide a configuration change or create an
unbounded background process. A separate confirmation suite evaluates the
selected candidate after the loop to reduce evaluator overfitting.

The normal v1 selection policy is lexicographic:

1. pass hard qualifiers;
2. maximize the primary objective;
3. preserve secondary objectives; and
4. prefer smaller, faster, or cheaper candidates when quality is effectively
   tied.

Pareto views may expose real tradeoffs among qualified candidates.

Adapter merging generates derived candidates and never promises improvement.
The initial merge workflow requires LoRA adapters with the same base-model and
tokenizer revisions, compatible target modules, integrity-verified parents, and
a maintained merge method. The resulting artifact records immutable parent
lineage and is evaluated against every parent before recommendation.

## 11. Local Store, Recovery, and Privacy

Canonical records live under a project-local .temper store. Hash-addressed
records are immutable; lifecycle events are append-only; derived state can be
rebuilt from events. Large bytes may live in configured local storage, but are
referenced by verified identities and lifecycle receipts.

Before invoking the runtime, Temper writes immutable experiment and resolved
runtime-request records. It captures public-safe logs, metrics, checkpoints,
artifact-ingestion receipts, failures, cancellation, and recovery events.
Interrupted runs retain evidence and can be resumed only when their retained
state is resume-compatible.

Canonical records and default exports must not contain secrets, private URLs,
local usernames, absolute paths, hostnames, process IDs, IP or MAC addresses,
or private artifact identifiers. Optional external references remain namespaced
and are redacted for public export.

Temper requires no hosted database or cloud account. Optional network imports
and model downloads are explicit ingress operations; after their inputs are
cached, training, evaluation, local use, replay, cleanup, and evidence browsing
remain available offline. Cross-boundary staging for a WSL worker is local and
must not make the worker's filesystem a second source of truth.

## 12. User Interface and Commands

The loopback UI and CLI expose the same application services. The UI never
writes canonical records directly. Normal navigation covers project setup,
dataset import, recipe resolution, preflight, run control, logs, metrics,
evaluation, playground review, recommendation, registry decisions, cleanup,
replay, local adapter use, verified export, loops, merge candidates, and
readiness assessment.

The CLI provides inspectable equivalents for status, manifest inspection,
verification, local-use launch, local batch inference, export verification,
replay planning, cleanup planning, and fixture workflows. It is not a second
product surface with different scientific semantics.

## 13. Explicit v1 Non-Goals

v1 does not:

- become a general ML platform or arbitrary training-script launcher;
- make an external trainer product the normal training path;
- expose linked or embedded external dashboards as the normal workflow;
- use model judges or a hidden universal quality score;
- become a general data editor, annotation system, chat client, or deployment
  control plane;
- create hosted inference endpoints or require a hosted Temper service;
- support unrestricted background optimization, distributed orchestration, or a
  public plugin marketplace;
- hard-lock confirmation data as a substitute for transparent evidence state;
- automatically promote, deploy, or mutate production artifacts; or
- commit private data, model weights, checkpoints, or private operational
  identifiers to the public repository.

## 14. v1 Acceptance Criteria

The architecture is successful when a clean checkout can demonstrate a
synthetic end-to-end project that:

1. imports and freezes a deterministic LLM dataset version;
2. resolves a versioned recipe into an inspectable manifest;
3. runs a Temper-owned adapter-runtime fixture path with canonical evidence;
4. verifies the resulting artifact even when quality evaluation is disabled;
5. evaluates applicable baselines and an explicit policy;
6. records playground or blind-review evidence with honest confidence labels;
7. retains, inventories, and safely cleans heavy bytes without deleting
   canonical evidence;
8. creates strict and adapted replay plans without conflating them;
9. runs a bounded iteration loop and evaluates its selected candidate with a
   confirmation suite;
10. creates and evaluates a compatible merged LoRA candidate;
11. loads a verified selected adapter for focused local interactive and batch
    inference, preserving settings and provenance when saved;
12. exports a verified adapter bundle without implying deployment; and
13. records a deployment-readiness assessment without controlling deployment.

The real-hardware acceptance path must also train, evaluate, and use a small
supported adapter through the Windows-host/WSL2 ROCm execution topology.
Native Windows execution is accepted only for combinations that pass the same
capability, lifecycle, evidence, and artifact contracts.

Every result must trace to immutable task, dataset, recipe, manifest, runtime,
base-model, tokenizer, evaluation-policy, artifact, and decision identities.
