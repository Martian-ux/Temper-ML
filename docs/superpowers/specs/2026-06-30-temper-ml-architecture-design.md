# Temper ML Architecture and Reuse Design

**Status:** Approved direction, amended after unified-product architecture review, implementation not started
**Date:** 2026-06-30
**Repository:** `Martian-ux/Temper-ML`

## 1. Product Thesis

Temper ML is a unified, local-first browser workbench for model experimentation.
It owns the supported visible workflow for creating projects, importing and
validating datasets, configuring experiments and training, running jobs,
tracking status, comparing candidates, evaluating outputs, conducting blind
review, registering artifacts, issuing recommendations, and managing narrow
local sweeps.

Temper's central product value is rigorous, composable evaluation inside one
coherent product. It should make it difficult to mistake a plausible-looking
result for a demonstrated improvement, and it should not ask a normal user to
assemble a workflow from MLflow, Optuna Dashboard, Argilla, DVC Studio,
Axolotl, LLaMA Factory, Noah, or another external interface.

Temper owns the user experience, product workflow, canonical records, and
evaluation evidence. External systems should normally be consumed as libraries,
headless services, command-line engines, runtimes, or storage providers. Their
dashboards and domain models do not define Temper's interface. Temper may
recreate a narrow visible feature when that is simpler and more coherent than
exposing a general-purpose external application.

Temper owns:

- projects and project policy;
- dataset import, validation, immutable version identity, and validation
  evidence;
- experiment and training configuration;
- run history, status, logs, normalized failures, interruption evidence, and
  retry evidence;
- native metric storage sufficient for supported workflows;
- charts, comparison views, evaluation suites, explicit gates, and
  recommendations;
- blind review for the initial supported scale;
- artifact registration and lifecycle events;
- basic local sweep management after the single-run path is proven;
- local execution controls and diagnostics;
- all supported navigation and workflow pages.

Temper does not attempt complete feature parity with MLflow, DVC, Optuna,
Argilla, Prefect, Axolotl, LLaMA Factory, Noah, or any other mature project.

## 2. Chosen Approach

Temper is a standalone repository and product. Noah is its first compatibility
workflow and known-good reference system, not Temper's architectural foundation
and not a permanent training-provider decision.

The dependency direction is:

```text
Temper UI
  -> Temper application services
    -> Temper domain and canonical store
      -> Temper-owned ports
        -> headless providers, libraries, runtimes, storage, and compatibility adapters
```

Temper core never imports a Noah recipe, assumes Noah dataset fields, treats an
external service as canonical, or delegates normal navigation to an external
dashboard. A separate process does not imply a separate product experience:
Temper may invoke a headless backend while still owning records, errors, charts,
state, and workflow.

The GitHub repository is the canonical development source of truth. A clean
checkout of a pushed branch or commit must contain everything required to
understand, modify, test, and review Temper without access to a maintainer
workstation or untracked local files.

### Integration Hierarchy

When introducing an external project, Temper uses the least product-fragmenting
and lowest-maintenance integration that satisfies the supported behavior:

1. Use it as a normal library.
2. Use a stable local API.
3. Invoke a pinned CLI or subprocess.
4. Extract or adapt a narrowly scoped subsystem.
5. Maintain a full fork only when other approaches are inadequate and the
   maintenance burden is justified.

Provider-specific configuration remains in the provider adapter unless it
represents a genuinely shared product concept. External dashboards may exist as
optional developer or diagnostic tools, but they are not part of the supported
user workflow.

### Rejected Alternatives

**Full custom rewrite:** Reimplementing trainers, tracking systems, standard
metrics, quantization, storage engines, or inference runtimes would delay the
product while creating maintenance work outside Temper's differentiator.

**Thin dashboard or launcher:** Directly exposing MLflow, DVC, trainer,
Optuna, and annotation dashboards would fragment identity and workflow. Temper
must present one project model and one user experience.

**Noah renamed as Temper:** Noah-specific fields, evaluation assumptions,
trainer choices, and promotion policy would leak into core and make later
workloads expensive or misleading.

## 3. Product Boundaries

### Native Temper Features

Temper natively implements deliberately narrow versions of features central to
its identity:

| Native feature | v0.1 intent |
| --- | --- |
| Canonical records | Project, dataset, task, experiment, run, artifact, evaluation, review, registry, and sweep records |
| Run history | Local run attempts, lifecycle events, status, retry/interruption evidence, logs, and normalized failures |
| Metrics and charts | Metric records and chart-ready views sufficient for the supported vertical slice |
| Candidate comparison | Baseline/candidate comparison through Temper evaluation records and UI |
| Dataset evidence | Dataset validation, immutable version identity, source evidence, and excluded-row evidence |
| Evaluation and gates | Eval Packs, suites, explicit gates, recommendations, and limitations |
| Blind review | Packet identity, leak audit, sealed judgments, reveal event, and initial-scale reviewer workflow |
| Artifact registry | Registration and lifecycle events over immutable artifact identities |
| Local execution | Launch, cancellation, status, diagnostics, retry, and recovery evidence |
| Basic sweeps | Explicit parameter lists, grid/random scheduling, trial queue/status/cancellation, and trial comparison |
| Browser interface | One coherent loopback application over application services |

### Reused Engines and Algorithms

Temper continues to reuse difficult, mature, non-differentiating capabilities:

| Capability | Likely reuse mode |
| --- | --- |
| PyTorch, Transformers, PEFT | Runtime/library dependency |
| Existing quantization implementations | Runtime/library dependency |
| Trainer runtimes including current Noah trainer, Axolotl, Unsloth, LLaMA Factory, or TRL | Headless provider behind a Temper training contract when justified |
| Standard evaluation metrics | Library dependency or narrow adapter |
| Optuna samplers and pruning algorithms | Hidden optimization engine only when advanced optimization is introduced |
| Sentence-transformer or similar embedding models | Runtime/library dependency for disclosed evaluators |
| ONNX Runtime | Runtime dependency for future compact-model execution or export |
| Storage SDKs and database engines | Storage provider/runtime dependency |
| Model-serving runtimes | Headless runtime behind Temper inference/evaluation ports |
| MLflow | Optional subordinate compatibility and tracking adapter |

Temper may expose its own optimization UI while using Optuna internally. It may
expose its own training configuration while invoking a headless training
backend. It may display its own metrics while optionally mirroring records to
MLflow.

### Noah Components That Must Remain Noah-Specific

- `ai_draft` and `your_revision` field interpretation;
- rewriting and humanization instructions;
- public-humanizer dataset import and filtering recipes;
- prose-only generation assumptions;
- the fixed seven-prompt evaluation cycle;
- fixed entity-preservation counts and thresholds;
- adapter-strength sweep policy;
- Noah's current default adapter and promotion policy;
- guessed style, rating, notes, or tags in the Noah registry;
- production mutation guards specific to the Noah application.

These belong to a Noah compatibility adapter, recipe, task definition, Eval
Pack, and project policy.

## 4. Canonical Domain Model

The complete vocabulary is defined now, but v0.1 implements only behavior
exercised by the supported product slice. The vocabulary is not permission to
create speculative modules, a public plugin marketplace, or empty adapters.

### Project and ProjectPolicy

A project is a stable local workspace with generated identity, display
metadata, project policy, canonical store, and configured integration
references. `ProjectPolicy` is an immutable, hash-addressed revision of
decisions that affect interpretation or lifecycle, including gates, review,
reveal, baseline, retention, and provider capability policy.

### Dataset and DatasetSchema

A dataset is a logical collection. Each imported state becomes an immutable
version. A dataset version records source evidence, normalized schema, splits,
record counts, content hashes, validator identity/configuration, validation
results, excluded-row evidence, and provenance. Dataset version identity is
derived from canonical metadata and content hashes, never absolute file paths.

### TaskDefinition

A task definition maps dataset fields into inputs, expected outputs, rendering
rules, objectives, and required backend capabilities. Core does not assume
prose, rewriting, or Noah source/target field names.

### Experiment and Training Request

An experiment is an immutable scientific intention represented by a resolved
manifest. It binds project/policy revision, dataset and task revisions, base
model and tokenizer identities, canonical training request, baseline/candidate
definitions, evaluation suite revision, code/environment evidence, and requested
optional integrations.

The canonical training request contains shared product concepts only. Provider
switches should not require rewriting project, evaluation, review, artifact, or
registry records.

### Run and Provider Evidence

A run is one execution attempt of an experiment, evaluation, review, or sweep
trial. It has immutable request and provider-request records, hash-linked
lifecycle events, derived state, logs, normalized failures, and terminal result
evidence. Retries create new run IDs and never overwrite prior attempts.

Each training provider accepts a fully resolved canonical training request and
returns normalized evidence:

- provider identity and version;
- exact resolved provider request;
- code revision and environment summary;
- dataset, base-model, and tokenizer identities;
- training state events;
- logs and normalized failures;
- output artifact descriptors and hashes;
- metrics and completion status.

### ModelArtifact and Registry

A model artifact is an immutable model-bearing output or reference. It records
artifact kind, content hash or immutable upstream revision, producing run,
parent artifacts, base-model compatibility, task/runtime compatibility,
storage references, provenance, and creation evidence.

The registry is an append-only event stream and derived view over immutable
artifacts. Registration, recommendation, promotion, deprecation, and retirement
are explicit lifecycle events. Registration never implies promotion.

### Evaluator, EvalPack, EvaluationSuite, and Gate

An evaluator is one versioned measurement instrument. An Eval Pack is an
immutable distribution of evaluators and evidence required to interpret them.
An evaluation suite composes Eval Pack evaluators, comparison variants,
aggregation rules, and gates. A gate is a simple, explicit rule that returns
`passed`, `failed`, or `inconclusive`; it performs no side effects.

Suites may combine deterministic checks, standard metrics, compact evaluators,
optional larger judges, and blind human review. Every component result is shown
before any disclosed aggregate. Temper never presents an unexplained universal
quality score.

### HumanReview

Human review records blind packet hash, item ordering, variant aliases, review
instructions, judgments, reviewer provenance, and reveal state. Packet creation,
leak audit, judgments, and reveal are immutable lifecycle records. Review-facing
payloads cannot expose internal mappings before judgments are sealed.

### Sweep

A sweep is a Temper-owned set of trial records over a frozen experiment family.
The initial native sweep feature supports explicit parameter lists, grid search,
random search, trial queue, trial status, trial cancellation, and trial
comparison through normal Temper evaluation. It does not recreate advanced
Optuna samplers, pruning algorithms, distributed studies, or multi-objective
optimization infrastructure. When advanced optimization becomes necessary,
Temper may use Optuna as a hidden engine while retaining Temper-owned trial
identities, records, UI, evaluation, and recommendations.

### ExternalReference

An external reference links a canonical Temper record to another system. MLflow
run IDs, DVC revisions, trainer job IDs, lab-server handles, and similar values
are external references only; they never replace Temper IDs or hashes.

## 5. Identity and Source-of-Truth Rules

Temper distinguishes logical identity, execution identity, and content
identity:

| Record | Logical/execution identity | Content identity |
| --- | --- | --- |
| Project | Generated `project_id` | Not applicable |
| Project policy | Stable policy ID | Policy-revision hash |
| Dataset | Generated `dataset_id` | Dataset-version hash |
| Task definition | Stable task name/ID | Task-revision hash |
| Experiment | Generated `experiment_id` | Manifest SHA-256 |
| Run/trial | Generated attempt ID | Request, provider-request, result, and event-head hashes |
| Model artifact | Generated `artifact_id` | Byte, bundle, or immutable upstream-revision identity |
| Eval Pack | Stable pack ID | Pack-revision hash |
| Evaluation suite | Stable suite ID | Suite-revision hash |
| Evaluation execution | Generated evaluation run ID | Result hash |
| Human review | Generated review ID | Packet, audit, judgment-set, and reveal hashes |
| Registry | Project registry ID | Append-only event-head hash |
| Sweep | Generated sweep ID | Frozen sweep definition hash |

Every content identity names a projection version. Record schema versions and
identity-projection versions are separate. The domain prefix prevents equal JSON
from different record types sharing an identity accidentally.

External references are metadata attached to canonical records. Temper remains
fully inspectable and usable when an optional external service is unavailable.

## 6. Canonical Local Store

The v0.1 project keeps canonical metadata under `.temper/`:

```text
project.temper.json
.temper/
  datasets/<dataset-id>/<version-hash>/dataset.json
  tasks/<task-id>/<revision-hash>/task.json
  policies/<policy-id>/<revision-hash>/policy.json
  experiments/<experiment-id>/<manifest-sha256>/manifest.json
  experiments/<experiment-id>/index.json
  runs/<run-id>/request.json
  runs/<run-id>/provider-request.json
  runs/<run-id>/result.json
  runs/<run-id>/events/<sequence>-<event-hash>.json
  runs/<run-id>/state.json
  runs/<run-id>/logs/
  artifacts/<artifact-id>/artifact.json
  eval-packs/<pack-id>/<revision-hash>/pack.json
  suites/<suite-id>/<revision-hash>/suite.json
  evaluations/<evaluation-run-id>/
  reviews/<review-id>/
  sweeps/<sweep-id>/sweep.json
  sweeps/<sweep-id>/trials/<trial-id>/trial.json
  registry/events/<sequence>-<event-hash>.json
  registry/state.json
```

Large datasets, checkpoints, and models may remain outside the metadata store.
Canonical records reference them by content hash and storage reference. Readers
verify hash-addressed records and bundle members. Mutable index/state files are
derived conveniences and are never scientific evidence.

## 7. Adapter and Dependency Boundaries

Temper-owned ports are narrow and capability-oriented. Adapters depend on
Temper ports; Temper domain code never depends on adapter implementations.
Every adapter translates canonical requests into fully resolved provider
requests and stores that translation before execution.

### TrainingProvider Contract

A training provider must:

- describe supported task and artifact capabilities;
- validate a canonical training request;
- resolve a complete provider request;
- launch, observe, cancel when supported, and recover or report interruption;
- collect logs, metrics, artifacts, and normalized failures;
- return typed outputs and evidence without mutating project policy or registry
  promotion state.

Provider-specific fields live in a namespaced adapter record. They do not become
Temper domain fields unless they are genuinely shared concepts.

### EvaluationBackend and InferenceRuntime

Evaluation backends validate suite requirements, resolve evaluators and runtime
requirements, launch or resume evaluation, and return typed raw results and
provenance. Inference runtimes load content-identified artifacts and generate or
predict through a task contract. Temper owns the visible evaluation, comparison,
review, and inference workflow even when the runtime is external.

### TrackingService and MLflow

MLflow is an optional subordinate compatibility and tracking adapter. Temper
v0.1 must not depend on the MLflow dashboard. Temper natively shows required run
history, metrics, charts, logs, failures, evaluation summaries, artifact
references, and recommendations. MLflow may optionally receive mirrored
parameters, metrics, artifact references, run references, and evaluation
summaries. Temper remains authoritative and fully usable while MLflow is stopped
or unavailable.

## 8. Noah Compatibility Boundary

Noah remains the first compatibility workflow and known-good reference system.
Temper separates the Noah workflow from any particular Noah training
implementation.

```text
Noah workflow
├── Dataset recipe
├── Task definition
├── Eval Pack
├── Baseline and project policy
└── Training provider
    ├── Existing Noah Transformers/PEFT trainer
    ├── Axolotl
    ├── Unsloth
    └── LLaMA Factory
```

The Noah workflow consists of:

1. **Noah dataset recipe:** Recognizes Noah-compatible fields and invokes or
   recreates the required preparation and filtering behavior.
2. **Noah task definition:** Defines rewrite inputs, outputs, instruction
   rendering, and model capabilities.
3. **Noah Eval Pack:** Wraps baseline/candidate evaluation, fixed prompts,
   entity checks, style checks, blind-review preparation, and documented
   limitations.
4. **Noah baseline and project policy:** Defines the frozen reference baseline,
   recommendation thresholds, review behavior, artifact policy, and the rule
   that v0.1 produces a recommendation without mutating Noah production.
5. **Training provider:** Selected independently through the backend bake-off.

Noah's current Transformers/PEFT trainer remains the fallback and parity
reference unless another provider clearly performs better through measured
evidence. It is not permanently authoritative merely because it already works.

Temper stores both its canonical request and the exact provider-side
translation. Noah-specific fields and thresholds remain confined to the Noah
recipe, task definition, Eval Pack, adapter, and project policy.

## 9. Training Backend Bake-Off and Selection Policy

Before committing to the long-term default real training backend, Temper runs a
bounded backend comparison using one frozen workload and an equivalent training
contract.

At minimum, compare:

1. existing Noah Transformers/PEFT trainer;
2. Axolotl;
3. Unsloth Core, if the current AMD/ROCm environment supports the required
   workflow reliably;
4. LLaMA Factory only if setup and parity requirements remain reasonably
   bounded.

The benchmark controls, as closely as each backend permits: dataset snapshot,
train/validation split, rendered training text, tokenizer revision, chat
template, sequence length, LoRA target modules, LoRA rank, LoRA alpha, LoRA
dropout, quantization mode, optimizer, learning-rate schedule, effective batch
size, gradient accumulation, seed, optimizer steps, evaluation frequency,
checkpoint frequency, base-model revision, and precision settings.

The comparison measures: peak VRAM, host RAM, swap usage, tokens or samples per
second, setup and model-load time, total wall time, training stability,
validation loss, final Temper evaluation results, blind-review results when
practical, adapter compatibility, reproducibility across repeated runs,
cancellation behavior, recovery behavior, quality of emitted logs and evidence,
difficulty of integration, and provider-specific code Temper must maintain.

Temper must not declare a backend better based only on advertised benchmarks.
The default provider is selected using measured performance, memory use, output
quality, reproducibility, ROCm reliability, evidence capture, integration
complexity, and maintenance burden. Temper may support multiple providers
through explicit internal adapters without building a generalized public plugin
marketplace.

## 10. Compact Evaluator Models and Task-Specific Models

Small, narrowly scoped model training remains a future product direction outside
v0.1. Temper should not call this TinyML unless targeting microcontrollers or
similarly constrained embedded systems. Preferred terms include **Compact
Evaluator Models**, **Task-Specific Models**, and **Evaluator Studio**.

Possible future use cases include style classification, tone classification,
entity-preservation detection, hallucination-risk detection, pairwise ranking,
quality classification, regression scoring, routing, safety classification,
dataset filtering, and small local evaluators packaged into Eval Packs.

A later workflow may define a narrow task and schema, import or label data,
train a compact model, evaluate held-out data, calibrate thresholds, record
intended domain and limitations, version the model as a Temper artifact, package
it into an Eval Pack, optionally export it to ONNX or another compact format,
and use it as one disclosed evaluator within a larger suite.

Compact evaluators must not produce unexplained universal quality scores. They
require explicit task definitions, versioned datasets, held-out evaluation,
calibration evidence, declared limitations, maturity status, versioned artifact
identity, and clear disclosure when used in a suite. This direction comes after
core training, evaluation, review, registry, and the unified interface are
proven.

## 11. v0.1 Vertical Slice and Sequencing

The first release supports one coherent local Temper workflow. The graphical
workflow appears early with fixture-backed behavior before real hardware
integrations are complete.

Recommended sequence:

1. Repository scaffolding and public-safe setup.
2. Canonical identity and persistence primitives.
3. Deterministic fixture walking skeleton.
4. Minimal fixture-backed Temper UI.
5. Training provider contract.
6. Backend bake-off and parity testing.
7. Noah compatibility contracts.
8. Selected real training provider integration.
9. Real evaluation, review, and inference integration.
10. UI expansion and operational hardening.
11. Basic sweeps.
12. Additional providers or integrations only when justified.

The early UI demonstrates create/open project, import a synthetic fixture
dataset, validate it, freeze an experiment, launch a deterministic fixture run,
inspect run status and metrics, compare baseline and candidate, open a
blind-review packet, register a synthetic artifact, and view an evaluation
recommendation. It must use application services and must never write canonical
records directly.

### v0.1 Constraints

- one local user and one local project at a time;
- Noah as the first compatibility workflow;
- selected real training provider must earn default status through bake-off;
- adapter artifacts only for training output;
- one Noah Eval Pack and one blind-review flow;
- explicit all-of gates with `passed`, `failed`, or `inconclusive` outcomes;
- optional subordinate MLflow mirroring only;
- local browser application served on loopback;
- basic sweeps only after the single-run workflow is proven;
- no cloud account or hosted control plane;
- no generalized plugin discovery or public dynamic plugin loader;
- no automatic production promotion;
- no compact-model training in v0.1.

## 12. Failure Handling and Recovery

- Temper writes the canonical experiment manifest, run request, and resolved
  provider request before invoking an external backend.
- Immutable requests, provider requests, completed results, event entries, and
  review evidence are never rewritten by later status.
- External invocation evidence is public-safe and excludes private paths,
  hostnames, usernames, PIDs, IP addresses, private artifact IDs, and private
  retrieval URIs.
- Metadata locks cover only atomic event/state transactions and are released by
  the operating system when a process exits.
- Startup verifies event sequence, predecessor hashes, and idempotency keys,
  then rebuilds derived state before accepting new events.
- Interrupted runs retain append-only events and may be marked interrupted or
  resumed when the provider supports it.
- Retry always creates a new run attempt.
- Missing MLflow does not invalidate an otherwise complete canonical run.
- Dataset, artifact, or manifest hash mismatch fails closed.
- Invalid Eval Packs or suites fail before model execution.
- Partial evaluation results remain diagnostic evidence but cannot satisfy a
  gate.
- Mechanical leak audit must pass for every reviewer-facing representation.
- Registry writes are atomic and single-writer.
- No failed adapter call can silently mutate project policy or production state.

## 13. Deferred Integrations

Integrations are added only after a demonstrated limitation. The table describes
likely reuse mode; it does not imply Temper will expose or embed external
dashboards.

| Integration | Likely reuse mode | Justifying condition |
| --- | --- | --- |
| Axolotl | Headless training provider | Bake-off shows bounded setup, better capability, efficiency, or maintainability |
| Unsloth Core | Library/headless training provider | ROCm workflow is reliable and measured results justify integration |
| LLaMA Factory | Headless training provider | Setup and parity remain bounded and coverage materially improves the product |
| TRL | Library/headless provider | Preference or reinforcement workflows become an approved product slice |
| Optuna | Hidden optimization engine | Native basic sweeps become insufficient and advanced samplers/pruning are needed |
| MLflow | Optional compatibility/tracking adapter | Mirroring helps diagnostics or interoperability without becoming authoritative |
| DVC | Storage/lineage provider or narrow adapter | Native hashes and storage references cannot meet a proven dataset need |
| Git LFS | Storage provider | Versioned large public files must live with repository history and costs are acceptable |
| Pandera | Validation library | Tabular validation exceeds native schema clarity or performance |
| Hydra | Configuration library | Configuration composition creates demonstrated duplication or unsafe overrides |
| Hugging Face Evaluate | Metric library | A maintained metric is better reused than implemented natively |
| lm-evaluation-harness or Lighteval | Headless evaluation engine | Standard benchmark coverage becomes a supported workflow |
| Sentence Transformers | Runtime/library dependency | A calibrated semantic evaluator is approved |
| Argilla or annotation platform | Optional external workflow/adapter | Native blind review cannot support required reviewer scale |
| Prefect | Headless orchestration engine | Local execution controls become insufficient for demonstrated multi-step workloads |
| ONNX Runtime | Runtime dependency | Compact evaluators or exported non-generative models need optimized local execution |
| Microsoft Olive | Optimization engine | Measured deployment optimization needs exceed direct ONNX/export tooling |
| Automatic artifact cleanup | Native lifecycle feature using storage providers | Retained bytes create observed disk pressure that explicit manual inventory cannot manage safely |

Temper may recreate a narrow visible feature when doing so creates a simpler,
more coherent product while continuing to reuse difficult algorithms and
infrastructure underneath. A full fork is a last resort.

## 14. Explicit Non-Goals

v0.1 will not:

- completely reimplement MLflow;
- completely reimplement Optuna;
- completely reimplement DVC;
- completely reimplement Argilla;
- completely reimplement Prefect;
- completely reimplement Axolotl or LLaMA Factory;
- expose a collection of linked or embedded external dashboards;
- build a generalized plugin marketplace;
- include a public dynamic plugin loader;
- build a distributed orchestration platform;
- create team accounts, billing, permissions, or a hosted control plane;
- implement an advanced optimizer from scratch;
- become a general annotation platform;
- create a universal evaluator score;
- train compact task-specific models;
- perform automatic production promotion;
- support every trainer, model type, task, or artifact kind;
- commit secrets, private datasets, licensed model weights, generated
  checkpoints, or unsuitable large binary artifacts to Git;
- implement abstractions not exercised by a supported product slice.

## 15. Repository, Cloud, and Operator Ownership

The canonical remote is `https://github.com/Martian-ux/Temper-ML`. `main`
remains the integration branch. No source, test, document, commit message, or
fixture may encode a maintainer's local checkout path or other non-public
machine identity.

The GitHub repository contains application source, schemas, tests, small
synthetic fixtures, documentation, handoff contracts, root setup instructions,
pinned dependencies, repository-owned setup/maintenance scripts, and sample
configuration with secret names but no secret values. No supported development
or review workflow may depend on an untracked local script, absolute workstation
path, unpushed commit, private notes, or already-configured Noah checkout.

The Noah compatibility boundary is represented by versioned schemas, contract
fixtures, and adapter tests. Real hardware validation is capability-gated and
cannot be claimed from a cloud-only fixture run.

## 16. Implementation Decomposition

The bounded v0.1 plan uses a small working vertical slice over speculative
breadth:

1. Establish public-safe repository scaffolding and setup.
2. Implement canonical identity and persistence primitives.
3. Build a deterministic fixture walking skeleton over application services.
4. Add the minimal fixture-backed browser UI early.
5. Define the Temper-owned training provider contract.
6. Run backend bake-off and parity tests.
7. Add Noah workflow contracts and fixtures.
8. Integrate the selected real training provider while preserving the fixture
   path.
9. Integrate real evaluation, review, and inference behind Temper UI.
10. Expand UI and harden local execution, diagnostics, cancellation, retry, and
    recovery.
11. Add native basic sweeps.
12. Add more providers or integrations only when a supported product slice
    justifies them.

Every file and abstraction in the implementation plan must support an
acceptance-tested behavior in the current stage. The first milestone remains
bounded and executable.

## 17. Acceptance Criteria

The architecture is successful when:

- Temper remains a standalone local-first product with one coherent browser
  workflow;
- normal users are not required to navigate external dashboards;
- the complete fixture flow is operable from one Temper project before real
  hardware integrations are complete;
- Noah remains the first compatibility workflow while its current trainer is
  only the fallback and parity reference;
- the selected real training provider earns its role through measured bake-off
  evidence;
- Temper can support multiple providers without becoming a public plugin
  marketplace;
- Temper remains inspectable and usable with MLflow stopped;
- native run history, metrics, charts, comparison, review, registry, and basic
  sweeps remain deliberately narrow;
- mature engines and algorithms are reused behind Temper-owned records and UI;
- compact task-specific model training is preserved for later and excluded from
  v0.1;
- every result traces to immutable dataset, manifest, code, model, evaluator,
  provider, artifact, and runtime identities;
- no Noah-specific field or fixed threshold appears in Temper core;
- provider-specific configuration is confined to adapters;
- interrupted or failed runs leave coherent local evidence;
- a blind review cannot open before leak audit or reveal before judgments are
  sealed;
- registering an artifact does not promote it or mutate Noah production;
- no section implies complete parity with an external tool or requires users to
  leave Temper for normal operation.
