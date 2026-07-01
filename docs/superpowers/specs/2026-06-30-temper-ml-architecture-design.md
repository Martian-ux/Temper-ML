# Temper ML Architecture and Reuse Design

**Status:** Approved direction, amended after architecture review, implementation not started
**Date:** 2026-06-30
**Repository:** `Martian-ux/Temper-ML`

## 1. Product Thesis

Temper ML is a local-first graphical workbench for training, comparing,
evaluating, reviewing, registering, and eventually exporting machine-learning
artifacts.

Temper's central product value is rigorous, composable evaluation. It should
make it difficult to mistake a plausible-looking result for a demonstrated
improvement.

Temper owns:

- the coherent product workflow and user experience;
- the canonical project and experiment domain model;
- immutable manifests, identities, hashes, and lineage;
- evaluation composition, evidence, limitations, and promotion policy;
- human-review artifacts and provenance;
- the registry of model artifacts and their lifecycle;
- project policy revisions and declarative evaluation gates;
- stable interfaces to training, evaluation, tracking, and inference systems.

Mature open-source tools and the current Noah stack provide specialized
capabilities behind those interfaces when integration is simpler and more
reliable than maintaining a native implementation.

## 2. Chosen Approach

Temper is a standalone repository and product. Noah is its first supported
workflow and compatibility integration, not its architectural foundation.

The dependency direction is:

```text
Temper UI
  -> Temper application services
    -> Temper domain and canonical store
      -> Temper-owned ports
        -> Noah and external-tool adapters
```

Temper core never imports a Noah recipe, assumes Noah dataset fields, or treats
an external service as the canonical source of truth.

The GitHub repository is the canonical development source of truth. A clean
checkout of a pushed branch or commit must contain everything required to
understand, modify, test, and review Temper without access to a maintainer workstation
or untracked local files.

For v0.1, the Noah integration should prefer existing command, file, and local
HTTP contracts over broad imports of `noah_style` internals. This keeps the
boundary observable and lets the existing workflow continue independently.
Generic behavior may be extracted from Noah into Temper only with
behavior-preserving contract tests and explicit ownership transfer.

### Rejected Alternatives

**Full custom rewrite:** Reimplementing trainers, tracking, standard metrics,
quantization, or inference runtimes would delay the product while creating
maintenance work outside Temper's differentiator.

**Thin dashboard:** Directly exposing MLflow, DVC, trainer, and annotation
dashboards would fragment identity and workflow. Temper must present one
project model and one user experience.

**Noah renamed as Temper:** Noah-specific fields, evaluation assumptions, and
promotion policy would leak into core and make later workloads expensive or
misleading.

## 3. Product Boundaries

### Temper-Owned Responsibilities

| Responsibility | Temper ownership |
| --- | --- |
| Project lifecycle | Project identity, settings, local paths, and policy |
| Project policy | Immutable policy revisions for gates, review, promotion, and retention |
| Dataset lineage | Logical dataset, immutable versions, schema, validation evidence |
| Task semantics | Inputs, outputs, objective, rendering, and compatible capabilities |
| Experiment definition | Immutable resolved manifest and reproducibility evidence |
| Run lifecycle | Attempts, status, events, logs, outputs, and external references |
| Artifact lineage | Byte identity, parentage, producing run, and compatibility metadata |
| Evaluation | Suites, Eval Packs, evaluators, results, evidence, and limitations |
| Human review | Blinding, packet identity, judgments, provenance, and reveal state |
| Registry | Registration and lifecycle events over immutable artifacts |
| Integration contracts | Backend, runtime, tracking, and storage ports |
| User experience | One local application spanning the complete workflow |

### Reused Capabilities

| Capability | Initial provider | v0.1 reuse mode |
| --- | --- | --- |
| QLoRA training | Noah Transformers/PEFT trainer | Invoke unchanged through `NoahTrainingBackend` |
| Noah dataset preparation | Noah dataset utilities and recipes | Invoke through a Noah recipe adapter |
| Baseline/candidate evaluation | Noah evaluator path | Wrap as the first Noah Eval Pack |
| Blind review packet | Noah experiment review | Invoke and import verified artifacts |
| Inference playground | Noah lab server | Connect through its local HTTP contract |
| Tracking and charts | MLflow | Optional subordinate tracking adapter |
| Canonical JSON and atomic persistence patterns | Noah experiment manifest code | Extract behavior into Temper core with parity tests |
| Locks and worktree evidence | Noah experiment manifest code | Extract behavior into Temper infrastructure |

### Noah Components That Must Remain Noah-Specific

- `ai_draft` and `your_revision` field interpretation;
- rewriting and humanization instructions;
- public-humanizer dataset import and filtering recipes;
- prose-only generation assumptions;
- the fixed seven-prompt evaluation cycle;
- the fixed entity-preservation counts and thresholds;
- adapter-strength sweep policy;
- Noah's default adapter and promotion policy;
- guessed style, rating, notes, or tags in the Noah registry;
- production mutation guards specific to the Noah application.

These belong to a Noah compatibility adapter, recipe, task definition, Eval
Pack, and project policy.

## 4. Canonical Domain Model

The complete vocabulary is defined now, but v0.1 implements only behavior
exercised by the first vertical slice. The vocabulary is not permission to
create an abstract plugin framework.

### Project

A stable local workspace with a generated `project_id`, display metadata,
project policy, canonical store, and configured integration references.

The project ID is stable and is not content-derived.

### ProjectPolicy

`ProjectPolicy` is an immutable, hash-addressed revision of project decisions
that affect interpretation or lifecycle. It includes evaluation-gate policy,
review and reveal requirements, promotion rules, baseline resolution,
retention labels, and allowed backend capabilities.

Changing policy creates a new revision. Experiments reference the exact policy
revision they use. Display metadata and maintainer notes do not participate in
policy identity.

### Dataset and DatasetSchema

`Dataset` is a logical collection. Each imported state becomes an immutable
dataset version.

`DatasetSchema` defines named fields, types, required values, constraints, and
record-level validation. It describes data shape without assigning universal
meaning to fields.

A dataset version records:

- source identity and revision when available;
- normalized schema;
- split names and record counts;
- content hashes for every source or materialized split;
- validator identity and configuration;
- validation results and excluded-row evidence;
- creation time and provenance.

The dataset version hash is derived from canonical metadata and content hashes,
never from absolute file paths.

### TaskDefinition

`TaskDefinition` maps named dataset fields into task inputs, expected outputs,
rendering rules, objective, and required backend capabilities.

It may define generation, classification, regression, ranking, structured
prediction, or another explicit contract. Core does not assume prose,
rewriting, or source/target field names.

The Noah task definition maps `ai_draft` and `your_revision` into Noah's
rewrite objective and prompt format.

### Experiment

An experiment is an immutable scientific intention represented by a resolved
manifest. It binds:

- project and policy revision;
- dataset version and task-definition revision;
- base model identity and revision;
- training preset and resolved backend configuration;
- baseline and candidate definitions;
- evaluation suite revision;
- code and environment evidence;
- requested tracking integrations.

`experiment_id` is a stable generated identifier. `manifest_sha256` is the
content identity of the immutable definition. Editing an experiment produces a
new manifest and hash under the same logical experiment ID.

### Run

A run is one execution attempt of an experiment or evaluation. It has a
generated `run_id` and references the exact manifest hash it attempts.

The run is represented by separate records:

- immutable `request.json`, written before external execution;
- immutable `provider-request.json`, containing the fully resolved adapter
  translation;
- immutable, ordered events recording observed lifecycle transitions;
- disposable `state.json`, derived from the verified event stream;
- immutable terminal `result.json`, written once and sealing log indexes,
  artifact identities, external references, and failure evidence by hash.

Retries create new run IDs. They do not overwrite prior attempts.

### ModelArtifact

A model artifact is an immutable model-bearing output or reference. Supported
artifact kinds may include adapter, checkpoint, full model, evaluator model,
tokenizer bundle, quantized model, or export bundle.

Each artifact records:

- a stable catalog `artifact_id`;
- artifact kind and format;
- content hash or immutable upstream revision;
- producing run and parent artifacts;
- base-model compatibility;
- task and runtime compatibility metadata;
- storage references;
- provenance and creation evidence.

The content hash identifies physical bytes. The artifact ID identifies the
catalog record. Identical bytes at different paths share physical identity but
may have distinct storage references.

An artifact selected for execution must be materialized as an immutable
verified snapshot for the duration of that execution. Directory bundles use a
canonical member projection and reject symlinks. A mutable external directory
may be used for discovery, but it cannot be loaded after verification as though
it were immutable. Large upstream base models may satisfy this rule through a
provider's immutable revision-addressed cache rather than being copied into the
Temper project.

### Evaluator

An evaluator is one versioned measurement instrument. It declares accepted
inputs, output schema, configuration, dependencies, determinism, resource
requirements, and limitations.

Evaluator types include:

- deterministic checks;
- conventional metrics;
- compact local evaluator models;
- optional large-model judges;
- human-review instruments.

### EvalPack

An Eval Pack is an immutable, versioned distribution of evaluators and the
evidence required to interpret them. It includes:

- evaluator descriptors and versions;
- model and asset identities;
- default configurations;
- calibration datasets or references;
- calibration results;
- known limitations and intended uses;
- reproducible execution requirements;
- result schemas.

The first Eval Pack is Noah-specific and wraps the existing evaluation path
without presenting its fixed assumptions as universal.

### EvaluationSuite

An evaluation suite is a project-selected composition of Eval Pack evaluators,
comparison variants, aggregation rules, and gates.

The suite makes every aggregate calculation inspectable. It never presents an
unexplained "AI quality score" as objective truth.

### Gate

A gate is an immutable declarative value object contained in an evaluation
suite. It identifies:

- the evaluator output or disclosed aggregate it reads;
- a simple comparison operator and expected value or threshold;
- the minimum evaluator maturity allowed to satisfy it;
- behavior for missing, errored, or non-finite input;
- whether the gate is required or advisory;
- whether failure blocks a recommendation or makes it inconclusive;
- a stable explanation shown with the result.

v0.1 gates are an explicit all-of list. Temper does not introduce a general
expression language. Gate results are `passed`, `failed`, or `inconclusive`;
they never perform promotion or another side effect.

### EvaluationResult

An evaluation result is an immutable execution output tied to:

- experiment manifest hash;
- suite and Eval Pack revisions;
- evaluator identities and configurations;
- runtime and model identities;
- inputs and variant ordering;
- raw per-item results;
- disclosed aggregation;
- logs, failures, and limitations.

Caller order is preserved in canonical results even when execution is grouped
for runtime reuse.

### HumanReview

Human review records the blind packet hash, item ordering, variant aliases,
review instructions, judgments, reviewer provenance, and reveal state.

Packet generation and judgments are immutable. Corrections create superseding
records rather than rewriting prior evidence.

The enforced lifecycle is:

```text
packet_created -> audit_passed -> review_open -> judgments_sealed -> revealed
```

The internal alias mapping may be created with the packet, but review-facing
APIs and files cannot expose it before an immutable judgment-set hash has been
sealed. Reveal is a separate explicit event. Automated leak auditing is
required before `review_open`.

### Registry

The registry is an append-only event stream and derived view over immutable
model artifacts.

Events may register, annotate, recommend, promote, deprecate, or retire an
artifact. Registry state is not part of artifact byte identity. Promotion is
always an explicit policy event and never an automatic consequence of
training.

Each event has a stream ID, sequence number, previous-event hash, event hash,
and idempotency key. One event is stored per immutable file. A derived registry
view records the event-head hash it consumed and can be rebuilt from the
verified chain.

### Backend

A backend performs specialized work through a stable Temper-owned port. A
backend reports capabilities, validates and resolves configuration, launches
work, emits observable status, and returns typed outputs.

v0.1 implements explicit Noah adapters. It does not implement dynamic plugin
discovery.

### Runtime

A runtime executes a model artifact for training, evaluation, or inference.
Its physical identity includes all construction-affecting inputs, including
model and tokenizer revisions, artifact hashes, dtype, quantization,
attention backend, device configuration, and software/runtime versions.

For the v0.1 ROCm path, captured construction evidence includes at minimum the
ROCm/HIP version, GPU model and architecture, Linux kernel version, PyTorch
version and build identifier, Python version, Transformers version, PEFT
version, Accelerate version, quantization backend and version, attention
backend and version, and relevant driver/runtime identifiers.

Paths alone never define runtime identity.

### ExternalReference

An external reference links a canonical Temper record to another system. It
contains the external system, reference kind, external ID, URI when useful,
and observation metadata.

MLflow run IDs, DVC revisions, trainer job IDs, and similar identifiers are
external references. They never replace Temper IDs or hashes.

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
| Run | Generated `run_id` per attempt | Request, provider-request, result, and event-head hashes |
| Model artifact | Generated `artifact_id` | Byte, bundle, or immutable upstream-revision identity |
| Eval Pack | Stable pack ID | Pack-revision hash |
| Evaluation suite | Stable suite ID | Suite-revision hash |
| Evaluation execution | Generated evaluation run ID | Result hash |
| Human review | Generated review ID | Packet, audit, judgment-set, and reveal hashes |
| Registry | Project registry ID | Append-only event-head hash |

### Identity Projection Rules

Every content identity names a projection version. Record schema versions and
identity-projection versions are separate.

`temper-canonical-json-v1` is UTF-8 JSON with lexicographically sorted object
keys, no insignificant whitespace, finite JSON numbers only, and a single
trailing newline. A structured identity is:

```text
SHA-256(
  UTF-8("temper:" + projection-name-and-version + "\n")
  + canonical-json(projected-fields)
)
```

The domain prefix prevents equal JSON from different record types sharing an
identity accidentally.

`temper-sha256-bytes-v1` hashes raw bytes without JSON projection. A directory
bundle projection sorts normalized POSIX relative paths, rejects absolute
paths, traversal, and symlinks, and hashes a canonical list of member role,
relative path, byte length, and `temper-sha256-bytes-v1` digest. Physical
location, mtime, ownership, permissions, and storage references are excluded.

| Content identity | Projection | Included fields | Excluded fields |
| --- | --- | --- | --- |
| Project policy revision | `project-policy-identity-v1` | Gate, review, reveal, promotion, baseline, retention, and required-capability policy | Policy ID, timestamps, display text, maintainer notes |
| Dataset version | `dataset-version-identity-v1` | Schema revision, split names, ordered content digests, counts, source revisions, validator identity/configuration, exclusions | Dataset ID, timestamps, absolute paths, storage references, display metadata |
| Task revision | `task-definition-identity-v1` | Input/output contracts, objective, rendering rules, required capabilities | Task ID, display name, timestamps, provider implementation |
| Experiment manifest | `experiment-manifest-identity-v1` | Project ID, policy hash, dataset/task revisions, base-model revision, canonical training request, baseline/candidate definitions, suite revision, code/environment evidence | Experiment ID, creation time, mutable status, storage paths, external run IDs |
| Run request | `run-request-identity-v1` | Run ID, manifest hash, canonical operation, selected backend capability/version, requested inputs | Status, later events, later timestamps, provider IDs |
| Provider request | `provider-request-identity-v1` | Run-request hash, adapter identity/version, fully resolved provider request and execution boundary | Provider response, mutable status, external run ID |
| Run result | `run-result-identity-v1` | Request/provider-request hashes, terminal outcome, artifact/result identities, sealed log index, normalized failure | Derived state, mutable paths, UI metadata |
| Model artifact bytes | `temper-sha256-bytes-v1` | Exact bytes | All metadata |
| Model artifact bundle | `model-artifact-bundle-v1` | Canonical member roles, relative paths, lengths, and byte digests | Storage root, mtimes, permissions, symlinks, external IDs |
| Immutable upstream model reference | `upstream-model-reference-v1` | Provider/repository identity, immutable revision, declared member roles and provider digests | Local cache path, download time, mutable branch/tag names |
| Physical runtime | `runtime-key-v1` | Model/tokenizer/artifact identities, construction settings, device/runtime configuration, and captured software versions | Virtual variant ID, adapter strength, paths, mtimes, process ID |
| Eval Pack revision | `eval-pack-identity-v1` | Evaluator descriptors, assets, defaults, calibration evidence, limitations, runtime requirements | Pack ID, timestamps, storage references |
| Evaluation suite revision | `evaluation-suite-identity-v1` | Eval Pack revisions, evaluator configs, variants, aggregation, gates | Suite ID, timestamps, execution results |
| Evaluation result | `evaluation-result-identity-v1` | Request identities, evaluator/runtime identities, inputs, caller order, raw results, disclosed aggregates, gate outcomes, limitations | Execution timestamps, storage paths, derived UI state |
| Blind packet | `blind-packet-identity-v1` | Manifest/suite references, prompt IDs, sources, anonymous labels, outputs, review instructions | Variant IDs, artifact paths, internal mapping |
| Internal blind mapping | `blind-mapping-identity-v1` | Packet hash, anonymous-label mapping, seed/procedure identity | Reviewer-facing paths, mutable reveal state |
| Leak audit | `blind-leak-audit-identity-v1` | Packet/rendered-payload hashes, allowlist revision, forbidden-token-set hash, findings | Unhashed diagnostic paths |
| Judgment set | `judgment-set-identity-v1` | Packet hash, reviewer provenance, ordered judgments, supersedes reference | Mapping, revealed identities, mutable notes |
| Registry/run event | `event-stream-entry-v1` | Stream ID, sequence, previous-event hash, idempotency key, event type/body, observed time | File path, derived state, lock metadata |

Projection definitions are part of the core compatibility contract. Changing
included or excluded fields requires a new projection version; it cannot be
smuggled into a record-schema change.

External references are metadata attached to canonical records. Temper remains
fully inspectable when an optional external service is unavailable.

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
  registry/events/<sequence>-<event-hash>.json
  registry/state.json
```

Large datasets, checkpoints, and models may remain outside the metadata store.
Canonical records reference them by content hash and storage reference.

Readers verify that every hash-addressed directory name matches the record's
declared identity projection and referenced bundle members. Event sequence
filenames use fixed-width decimal sequence numbers. Experiment `index.json`,
run `state.json`, and registry `state.json` are explicitly mutable, derived
conveniences and are never scientific evidence.

Persistence rules:

- manifests and completed results are immutable;
- lifecycle status uses hash-linked immutable events and a derived snapshot;
- writes are atomic;
- event sequence allocation and derived-state replacement use a kernel-backed
  advisory single-writer lock held only for that short transaction;
- training, evaluation, and inference never hold a metadata lock while model
  work executes;
- the operating system releases advisory locks when a process exits, including
  OOM, termination, or power-loss recovery; a leftover lock file is not a held
  lock and is harmless;
- PID-liveness checks, lock timeouts, and force-unlock controls are not used;
  startup verifies the event chain and rebuilds derived state instead;
- recovery ignores incomplete temporary files;
- hashes are verified when artifacts cross a boundary;
- paths are relocatable project references when possible;
- secrets never enter manifests, logs, or hashes.

## 7. Adapter and Dependency Boundaries

Temper-owned ports are narrow and capability-oriented:

### TrainingBackend

- describe supported task and artifact capabilities;
- validate canonical training requests;
- resolve a complete backend configuration;
- launch and observe a run;
- collect outputs and logs;
- report normalized failures and raw evidence.

`NoahTrainingBackend` launches the existing Noah trainer and records the Noah
repository commit, command, working tree evidence, environment, and resolved
trainer configuration.

For the first supported workstation, this adapter also owns the Windows-to-WSL
execution boundary. It records the WSL distribution and version, translates
paths explicitly, verifies that translated inputs resolve to the same content,
and reports the actual Linux command and environment. Outputs crossing from
Linux back to Windows receive a two-sided transfer receipt: the Linux-side
producer and Windows-side importer independently compute the declared
projection and must agree before Temper accepts the artifact. Windows or WSL
path spelling never participates in canonical artifact identity.

### EvaluationBackend

- validate suite requirements;
- resolve evaluators and runtime requirements;
- launch or resume evaluation;
- return typed raw results and provenance.

The v0.1 Noah implementation invokes the existing evaluator path and imports
its outputs into a Temper `EvaluationResult`.

The adapter exposes a versioned
`noah-evaluation-runtime-reuse-v1` capability. Support requires:

- content-derived `runtime-key-v1` identity over every construction-affecting
  model, tokenizer, adapter, quantization, attention, device, and software
  setting;
- exclusion of virtual variant ID and adapter strength from physical identity;
- one load per physical runtime key and no more than one resident quantized 14B
  runtime;
- a per-runtime lock covering reset strength, apply strength, format/tokenize,
  and generate;
- restoration of caller order after physical-key grouping;
- fail-closed resume matching manifest, prompts, evaluator/runtime schemas,
  decode configuration, requested variants, caller order, and runtime keys;
- result evidence containing runtime keys, load counts, and bounded peak-VRAM
  observations.

This capability describes the approved Noah runtime-reuse contract, which must
be implemented and verified in Noah before Temper's real-evaluation milestone.
The adapter explicitly reports unsupported capability when connected to the
legacy evaluator; Temper never claims optimized reuse in that state.

### InferenceRuntime

- report runtime capabilities;
- load a content-identified artifact;
- generate or predict through a task contract;
- expose status and normalized errors.

The v0.1 adapter uses the existing Noah lab server's local API. Temper does not
duplicate the inference runtime during the first slice.

### TrackingService

- create or associate an external run;
- log parameters, metrics, and artifacts;
- return external references;
- fail independently of canonical local persistence unless project policy
  explicitly requires tracking.

MLflow is the first implementation. Temper records the MLflow tracking URI and
run ID as external references.

### Adapter Rules

- Adapters depend on Temper ports; Temper domain code never depends on adapter
  implementations.
- Every adapter translates canonical requests into a fully resolved provider
  request and stores that translation.
- Provider-specific fields live in a namespaced extension object or adapter
  record, not generic core fields.
- An adapter cannot mutate registry promotion state.
- An adapter failure leaves enough local evidence to diagnose or resume work.
- Compatibility is explicit through capabilities and versioned contracts, not
  inferred from class names.

## 8. Evaluation and Compact Evaluator Models

Compact local evaluator models are important Temper instruments, not Temper's
entire identity.

Each model-based evaluator records:

- evaluator and Eval Pack revision;
- model and tokenizer identity;
- immutable model revision or content hash;
- input rendering and output interpretation;
- inference and runtime configuration;
- training and dataset provenance when available;
- calibration procedure and evidence;
- known failure modes and intended domain;
- agreement with held-out human judgments when available.

Model-based evaluators have an explicit maturity:

- **Experimental:** Useful for exploration; cannot independently gate
  promotion.
- **Calibrated:** Evaluated against a declared reference set; may contribute to
  a gate only with corroborating evidence.
- **Validated:** Meets a project-declared agreement and robustness standard on
  held-out human judgments; may gate within its documented domain.

Maturity is scoped to an evaluator revision and intended domain. It is not a
permanent label for a model family.

Suites may combine deterministic checks, standard metrics, compact evaluators,
optional larger judges, and blind human review. Results expose each component
before any aggregate.

Before a review becomes available, Temper audits both the canonical packet and
every rendered file or API payload presented to reviewers. The audit enforces
an exact field allowlist and scans for known variant IDs, artifact paths,
registry names, model labels, provider configuration, and internal mapping
values. Findings and the forbidden-token set are sealed in a hashed audit
record. Any finding fails closed.

This mechanical audit cannot prove semantic indistinguishability. Review
evidence therefore discloses that reviewers may still infer a model from its
behavior or writing style.

## 9. Noah Compatibility Boundary

Noah integration consists of seven explicit assets:

1. **Noah dataset recipe:** Recognizes Noah fields and invokes existing
   preparation and filtering behavior.
2. **Noah task definition:** Defines rewriting inputs, outputs, instruction
   rendering, and model capabilities.
3. **Noah training backend:** Translates a Temper training request into the
   current Transformers/PEFT QLoRA command.
4. **Noah baseline resolver:** Resolves the frozen v1 baseline from project
   policy and verifies its immutable artifact bundle and execution settings.
5. **Noah Eval Pack:** Wraps existing baseline/candidate evaluation, fixed
   prompts, entity checks, style checks, and blind-review preparation with
   documented limitations.
6. **Noah runtime-reuse capability:** Adapts the approved content-keyed,
   strength-safe two-load evaluation contract.
7. **Noah project policy:** Defines recommendation thresholds and confirms
   that v0.1 produces a recommendation without mutating Noah production.

The v0.1 `baseline-resolution-v1` policy freezes:

- the private project policy's immutable baseline artifact reference;
- adapter strength `0.25`;
- the `model-artifact-bundle-v1` digest of the verified adapter snapshot;
- base-model and tokenizer immutable revisions;
- chat-template, quantization, decoding, instruction, prompt-set, evaluator,
  and runtime projection identities.

The public source repository defines the baseline contract but does not contain
private operational artifact IDs or fingerprints. Those values live in the
untracked project policy or approved private artifact store. Public tests use
synthetic fixtures. The complete bundle projection is authoritative, and
resolution verifies the immutable snapshot immediately before loading and
fails closed. The mutable Noah registry may assist discovery but cannot change
the selected bytes, strength, or settings.

Temper stores both its canonical request and the exact Noah-side translation.
A Noah-side change requires a versioned adapter contract and a recorded Noah
commit.

Temper changes do not directly modify the Noah repository. Required Noah
changes are coordinated through a versioned contract, public base commit,
acceptance criteria, and expected output type.

## 10. v0.1 Vertical Slice

The first release supports one complete, real workflow:

1. Create or open a local Temper project.
2. Import one Noah-compatible dataset.
3. Validate it against the Noah schema and produce immutable validation
   evidence.
4. Select one supported base model, the existing QLoRA preset, and the
   policy-pinned verified v1 baseline.
5. Generate and freeze an immutable Temper experiment manifest.
6. Launch one real training run through `NoahTrainingBackend`.
7. Display canonical status, events, and logs in Temper.
8. Associate optional MLflow tracking without making it authoritative.
9. Compare candidate and baseline through the Noah Eval Pack.
10. Produce a blind-review artifact and pass automated mechanical leak audit.
11. Seal review judgments before revealing the internal mapping.
12. Register the resulting adapter as a Temper `ModelArtifact`.
13. Test the registered artifact through the existing Noah inference runtime.
14. Produce a promotion recommendation without mutating Noah production.

### v0.1 Constraints

- one local user and one local project at a time;
- one Noah-compatible generation task;
- one Noah Transformers/PEFT QLoRA backend;
- the current Windows host and WSL/ROCm execution path;
- adapter artifacts only for training output;
- one Noah Eval Pack and one blind-review flow;
- explicit all-of gates with `passed`, `failed`, or `inconclusive` outcomes;
- optional local MLflow integration;
- existing Noah inference runtime;
- local browser application served on loopback;
- no cloud account or hosted control plane;
- no generalized plugin discovery;
- no automatic production promotion.

The generic domain model is proven through this slice, but the UI presents the
supported path rather than disabled controls for future integrations.

## 11. Failure Handling and Recovery

- Temper writes the canonical experiment manifest, run request, and resolved
  provider request before invoking an external backend.
- Immutable run request and provider-request records are never rewritten by
  later status or results.
- Every external invocation records the resolved request, process identity,
  working directory, execution host, code revision, and logs.
- Metadata locks cover only atomic event/state transactions and are released by
  the operating system when a process exits.
- Startup verifies event sequence, predecessor hashes, and idempotency keys,
  then rebuilds derived state before accepting new events.
- Interrupted runs retain append-only events and may be marked interrupted or
  resumed when the backend supports it.
- Retry always creates a new run attempt.
- Missing MLflow does not invalidate an otherwise complete canonical run.
- Dataset, artifact, or manifest hash mismatch fails closed.
- Invalid Eval Packs or suites fail before model execution.
- Partial evaluation results remain diagnostic evidence but cannot satisfy a
  gate.
- Blind aliases cannot be revealed through the review artifact before an
  immutable judgment-set hash and explicit reveal event.
- Mechanical leak audit must pass for every reviewer-facing representation.
- Registry writes are atomic and single-writer.
- No failed adapter call can silently mutate project policy or production
  state.

## 12. Deferred Integrations

Integrations are added only after a demonstrated limitation:

| Integration | Justifying condition |
| --- | --- |
| Unsloth | Measurable training efficiency or hardware support unavailable in the current backend |
| Axolotl | Required model/training configuration is materially simpler or unsupported in the current backend |
| LLaMA Factory | A supported workflow benefits from its training coverage enough to justify another adapter |
| TRL | Preference or reinforcement workflows become an approved product slice |
| DVC | Dataset lineage or storage cannot be managed reliably with hashes and current storage references |
| Git LFS | Versioned large files must live with repository history and measured costs are acceptable |
| Pandera | Tabular validation exceeds the clarity or performance of native schema validation |
| Hydra | Configuration composition creates demonstrated duplication or unsafe overrides |
| Hugging Face Evaluate | A required maintained metric is better reused than implemented natively |
| lm-evaluation-harness | Standard language-model benchmark coverage becomes a supported workflow |
| Lighteval | Its backend or benchmark support materially reduces evaluation maintenance |
| Sentence Transformers | A semantic evaluator is approved with calibration requirements |
| Optuna | Manual experiment sweeps become an observed bottleneck |
| Argilla or another annotation platform | Native blind review cannot support the required reviewer scale or workflow |
| Prefect | Local orchestration and recovery are insufficient for demonstrated multi-step workloads |
| ONNX Runtime | Compact evaluators or exported non-generative models need a stable optimized runtime |
| Microsoft Olive | Measured deployment optimization needs exceed direct ONNX/export tooling |
| Automatic artifact garbage collection | Retained checkpoint and model bytes create observed disk pressure that manual inventory and explicit cleanup cannot manage safely |

v0.1 never deletes model or checkpoint bytes automatically. Artifacts from
failed or superseded runs remain referenced and are labeled `retained`,
`reclaimable`, or `pinned` by project policy. The status/dump command reports
their identity, provenance, references, and disk footprint. A later cleanup
feature must verify that no pinned run, evaluation, review, or registry record
depends on an artifact and must write a tombstone event; automatic garbage
collection remains deferred until the condition above is met.

For every proposed integration, architecture review asks:

1. Is the feature central to Temper's product identity?
2. Does a mature tool already solve it well?
3. Can it be wrapped without owning Temper's canonical model?
4. Is integration simpler than maintaining a small native implementation?
5. Does it improve the coherent user experience?

## 13. Explicit Non-Goals

v0.1 will not:

- reimplement a training engine, tracking dashboard, ONNX runtime,
  quantization algorithm, or standard metric;
- expose a collection of disconnected external dashboards;
- support every trainer, model type, task, or artifact kind;
- build a public plugin marketplace or dynamic plugin loader;
- generalize Noah's task-specific evaluator into an unexplained universal
  score;
- make MLflow, DVC, Hydra, or a trainer's IDs canonical;
- create cloud synchronization, accounts, teams, permissions, or billing;
- commit secrets, private datasets, licensed model weights, generated
  checkpoints, or other unsuitable large binary artifacts to Git;
- perform automatic checkpoint or model garbage collection;
- automatically promote or deploy an artifact;
- replace or destabilize Noah's current production workflow;
- extract code from Noah without parity tests and clear ownership;
- implement abstractions that the v0.1 vertical slice does not exercise.

## 14. Repository, Cloud, and Operator Ownership

The canonical remote is `https://github.com/Martian-ux/Temper-ML`. `main`
remains the integration branch. No source, test, document, commit message, or
fixture may encode a maintainer's local checkout path or other non-public
machine identity.

### Repository-Contained Development

The GitHub repository contains:

- all application and library source;
- schemas, identity-projection definitions, and migrations;
- unit, contract, integration-fixture, and UI tests;
- small deterministic datasets, model substitutes, and artifact fixtures
  needed by the walking skeleton;
- documentation, architecture decisions, implementation plans, and handoff
  contracts;
- root `AGENTS.md` instructions with authoritative setup, test, lint, and
  verification commands;
- version-pinned dependency declarations and lockfiles;
- repository-owned Codex cloud setup and maintenance scripts;
- sample configuration with secret names but no secret values;
- scripts and manifests that resolve public external artifacts by immutable
  revision and content hash;
- schemas and adapters that accept private artifact descriptors at runtime
  without committing private descriptor values.

No supported development or review workflow may depend on an untracked local
script, an absolute workstation path, an unpushed commit, a maintainer's private
notes, or an already-configured Noah checkout.

"Contained in GitHub" does not mean committing unsuitable model, dataset, or
checkpoint bytes. Large, private, licensed, or generated artifacts remain in
external content-addressed storage. For public artifacts, the repository may
contain complete descriptors, provenance, expected hashes, and retrieval
procedures. For private artifacts, the repository contains only schemas,
obviously synthetic placeholders, and small legal fixtures; actual IDs,
filenames, revisions, fingerprints, provenance, and retrieval details remain
in untracked project state or an approved private store.

The Noah compatibility boundary is represented in Temper by versioned schemas,
contract fixtures, and adapter tests. A real Noah integration may obtain the
public Noah repository at an explicitly pinned commit or use a configured
checkout, but ordinary Temper development and the fixture walking skeleton do
not require it.

### Codex Cloud Environment Contract

Codex cloud work starts from a pushed Git branch or commit. The environment
configuration invokes repository-owned scripts:

```text
setup:       bash scripts/codex-cloud-setup.sh
maintenance: bash scripts/codex-cloud-maintenance.sh
```

The scripts are idempotent, noninteractive, and safe in the default Linux
container. They install pinned dependencies, validate tool versions, prepare
small fixtures, and run a fast environment preflight. The maintenance script
updates an existing cached environment after checkout without depending on
state from an earlier commit.

The cloud contract also requires:

- documented and pinned Python, Node.js, and other selected runtime versions;
- a repository-root `AGENTS.md` that names exact verification commands;
- no secret requirement for ordinary setup, tests, documentation, or fixture
  execution;
- secret values configured only in the Codex environment when an optional
  setup-time private dependency requires them;
- tests that do not require agent-phase internet access;
- a narrowly allowlisted, read-only network policy only for explicitly marked
  integration tasks that need external public artifacts;
- Linux-compatible paths and commands for all cloud-supported work;
- capability-gated WSL/ROCm and GPU checks that skip with an explicit reason in
  cloud environments without that hardware;
- one command that proves the clean-checkout fixture workflow end to end.

Cloud compatibility has three verification tiers:

1. **Required offline tier:** Unit tests, identity fixtures, documentation
   checks, status/dump commands, and the deterministic walking skeleton.
2. **Required contract tier:** Noah adapter, MLflow, runtime, and artifact
   boundary tests using repository fixtures or local test doubles.
3. **Capability-gated hardware tier:** Real QLoRA, ROCm, 14B evaluation, and
   local inference. Their public execution contracts remain reproducible from
   repository fixtures, while real private descriptors are supplied at runtime.
   They run only where the declared hardware and external artifacts are
   available.

A cloud agent may complete and verify ordinary product work using tiers one and
two. Work affecting the hardware tier cannot be declared fully validated until
an appropriate local or GPU-capable environment reports the pinned commit and
evidence.

This contract follows the published Codex behavior for
[cloud environments](https://developers.openai.com/codex/cloud/environments),
[agent internet access](https://developers.openai.com/codex/cloud/internet-access),
and repository
[`AGENTS.md`](https://developers.openai.com/codex/guides/agents-md).

### Contribution Ownership

After this specification and the implementation plan are approved:

- `main` remains the integration branch.
- Product work uses feature branches created from an explicit commit available
  on `origin`.
- Cross-repository Noah changes use versioned contracts, acceptance criteria,
  and public commit references without exposing private runtime evidence.
- Any remote task references a commit available on
  `origin`; local-only branches and working-tree state are invalid handoff
  inputs.
- Concurrent contributors declare file ownership and do not revert unrelated
  work.

Implementation begins only after the bounded plan is approved.

## 15. Implementation Decomposition

The bounded v0.1 plan uses a walking skeleton followed by measured replacement
and hardening:

1. **Executable walking skeleton:** Create/open a project, import a fixture
   Noah dataset, freeze a manifest, write immutable run requests, execute a
   deterministic fixture backend, produce an evaluation result and audited
   blind packet, seal fixture judgments, register an artifact, and inspect the
   result through a minimal application-service boundary. This path uses the
   v1 projection machinery and immutable request/result boundaries from its
   first executable test. The same stage adds root `AGENTS.md`, pinned
   dependency files, Codex cloud setup/maintenance scripts, and one
   clean-checkout command that runs the fixture path without secrets or
   agent-phase network access.
2. **Identity, persistence, and diagnostic inspection:** Complete cross-platform
   and adversarial coverage for every v1 hash projection, immutable artifact
   materialization, hash-linked event recovery, derived-state rebuilding, and
   narrow `temper status` and `temper dump` commands. These commands expose
   canonical records, references, storage footprint, and normalized failures
   before GPU work begins. Both required cloud verification tiers run here.
3. **Real Noah training path:** Replace the fixture trainer with the versioned
   Noah dataset/training adapter, Windows/WSL transfer receipts, optional
   MLflow reference, real QLoRA execution, and verified artifact import while
   retaining the executable end-to-end path.
4. **Real evaluation and review path:** Integrate the verified v1 baseline,
   approved physical-runtime reuse contract, Noah Eval Pack, explicit gates,
   automated leak audit, sealed judgments, reveal control, and inference
   runtime.
5. **Graphical workflow and operational hardening:** Build the local browser
   workflow over the proven application services, then harden interruption,
   resume, concurrency, diagnostics, accessibility, and production-quality
   error handling.

Identity projections, immutable request/result boundaries, and fail-closed
verification are part of the first skeleton, not deferred correctness work.
Every stage retains an executable end-to-end path. Expensive execution may use
deterministic fixtures only until the named real-runtime stage.

The implementation plan must name exact files, tests, acceptance commands, and
inter-repository handoffs.

## 16. Acceptance Criteria

The architecture is successful when:

- the complete Noah v0.1 flow is operable from one Temper project;
- a clean pushed checkout in a Codex cloud environment can run setup and the
  complete deterministic fixture workflow without workstation files, secrets,
  GPU hardware, or agent-phase internet access;
- all code, tests, schemas, small fixtures, instructions, and development
  scripts required by cloud work are versioned in the GitHub repository;
- every public external artifact needed for optional real integration has a
  committed immutable descriptor, expected hash, and retrieval or import
  procedure;
- private artifact integration uses committed schemas and synthetic fixtures,
  while actual identifiers and fingerprints remain outside Git;
- every result traces to immutable dataset, manifest, code, model, evaluator,
  and runtime identities;
- every content identity has a named, testable projection version;
- Temper remains inspectable with MLflow stopped;
- `temper status` and `temper dump` can explain work before the graphical
  application exists;
- no Noah-specific field or fixed threshold appears in Temper core;
- provider-specific configuration is confined to adapters;
- an interrupted or failed run leaves coherent local evidence;
- a blind review cannot accidentally reveal candidate identity;
- every reviewer-facing payload passes mechanical leak audit and remains
  unrevealed until judgments are sealed;
- the frozen v1 baseline resolves to verified bundle bytes at strength `0.25`
  without consulting mutable registry state;
- optimized Noah evaluation proves the approved physical-runtime reuse
  capability rather than merely claiming it;
- registering an artifact does not promote it or mutate Noah production;
- adding a second backend or task would require a new adapter and recipe, not a
  rewrite of canonical records;
- a cloud-verified commit cannot be represented as hardware-validated until a
  capable environment attaches the required real-runtime evidence;
- the UI exposes one coherent workflow rather than external-tool navigation.
