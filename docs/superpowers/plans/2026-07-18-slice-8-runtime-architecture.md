# Slice 8 library-backed runtime architecture

**Status:** implemented candidate; publication and integration remain
maintainer-gated.

**Base:** `7d1f9d66c5a8591de02f414c1dd84f88d4c2ad31`

**Scope:** the adopted Slice 8 local adapter runtime only. Retention/replay
execution, automated experiment loops, LoRA merging, deployment readiness,
hosted orchestration, and external trainer compatibility remain later work.

## Architecture decision from issue 12

GitHub issue 12 asked whether runtime ownership should use an actor-style
design. Slice 8 uses the useful ownership property without adopting an actor
framework or rewriting stable records and deterministic application services.

One `SerializedRunController` owns ordered non-canonical lifecycle state for
each live request. One `SerializedResourceCoordinator` serializes declared
accelerator leases. Both are small in-process ports protected by locks, accept
idempotent exact replay, reject conflicting replay or sequence gaps, and can be
rebuilt from the immutable runtime request plus the durable message ledger.
They never become canonical authorities.

The run controller admits these terminal paths:

```text
created -> active -> artifact_ready -> completed
                  -> cancelling -> cancelled
                  -> disconnected -> reconnected -> active
                  -> interrupted
                  -> failed
```

A terminal message wins once. Later completion, cancellation, or failure
messages are rejected. Resource release occurs in the adapter's `finally`
boundary, including cancellation, interruption, worker failure, validation
failure, and artifact-transfer failure.

Before the application service performs its unused-run check, it acquires a
run-scoped OS lock backed by an immutable private claim. The lock spans record
freezing, worker execution, output persistence, and the durable terminal event.
A private resolution marker is written only when launch never began or exactly
one terminal path is the final durable event. A process that loses its OS lock
without that resolution leaves an unresolved claim that blocks relaunch; two
service or adapter instances therefore cannot both pass a read/check race and
start the same immutable run.

## Runtime layers

The public application-service seam remains `RunService` and
`LocalUseService`. The fixture adapter and library adapter both produce the
same normalized adapter member set and use the same canonical run, artifact,
availability, integrity, evaluation, selection, local-use, and export records.
No canonical schema change is required.

`LibraryAdapter` translates the existing frozen runtime request into the
narrow `LibraryBackend` port. The port supports capability probing, LoRA
training, checkpoint callbacks, heartbeat and control checks, evaluation
inference, focused inference, and batch inference. `LibraryInferenceRuntime`
re-verifies the normalized artifact and calls the same backend port for all
three inference operations. Private model and tokenizer source locations carry
the exact Temper base-model record reference and tokenizer content identity;
training and inference reject a source declaration that does not match the
frozen request or verified artifact. The adapter and inference runtime probe
the backend themselves and reject a caller-supplied capability or runtime
identity that does not match the observed target and library versions.

The implementations are:

- `DeterministicLibraryBackend`, a no-network, no-hardware contract double;
- `WslWorkerBackend`, the reference Windows/WSL process and transfer port; and
- `TransformersPeftBackend`, the real local library machinery and direct
  native-process implementation of the same port.

The real backend loads local tokenizer and model directories with remote code
disabled, constructs a PEFT LoRA model, prepares training and inference through
Accelerate, emits integer-microunit loss evidence, and saves an adapter payload
in safetensors or PyTorch-bin form. Its restricted `weights_only` checkpoint
captures adapter, optimizer, scheduler, mixed-precision scaler, prepared-loader
position, and CPU/device PyTorch RNG state. Restore prepares the same
Accelerator boundary, advances the deterministic loader to the exact next
batch, and then restores that state, so recovery follows the uninterrupted
training trajectory. Library-owned filenames and identifiers are normalized
before artifact ingestion and do not become Temper identity.

Checkpoint payloads bind their exact recipe resolution, decoded optimizer step,
and prepared-loader consumption count. Restore accepts only the complete
trajectory-preserving checkpoint schema and an integer step strictly below the
frozen training step count. A final-step checkpoint may be retained for audit
evidence, but it is explicitly non-resumable in both worker messages and
canonical receipts, so a disconnect after the final checkpoint cannot execute
an extra update.

## Windows/WSL boundary

The host freezes a private immutable `WorkerInvocation` containing the exact
request identity, operation, target class, resolved recipe, private worker
source locations, and content-identified input manifest. The launcher uses a
fixed argument vector equivalent to:

```text
wsl.exe --distribution <distribution> --exec <python> \
  -m temper_ml.runtime.worker_process --request <request>
```

It never invokes a shell. Distribution names, executable locations, portable
logical locations, and subject identities are validated before launch.

Inputs and outputs live under one explicitly configured shared root. Manifests
bind every logical member to a role, byte count, and SHA-256 content identity.
Writes are write-once; ingestion reopens and verifies every byte. Missing,
partial, replaced, linked, or corrupted members fail closed. Worker responses
are bound to the request, operation, run, and target, and output references
must remain inside that operation's output prefix with the declared role.
Input manifests must be host-to-worker, output manifests must be
worker-to-host, and every receipt reconstructs and matches the complete
manifest it claims to verify.

The worker durably writes one canonical JSON message file per sequence and a
subject-bound terminal response. The launcher reconstructs the controller from
those files before trusting a response. An exact completed response is reused
without spawning another process. A partial ledger requires reconciliation and
never causes a duplicate launch; any verified progress and checkpoints in that
ledger can be projected into an interrupted attempt. Cancellation, explicit
interruption, and timeout use write-once cooperative control markers. A host
message-handler failure also interrupts and terminates the child boundary.

The worker can probe, train, evaluate, and perform focused or batch inference,
but it has no canonical-store dependency. Only the Windows-side application
service appends canonical lifecycle evidence and ingests a verified artifact.

## Protocol and evidence rules

Typed messages cover launch, progress, metrics, checkpoints, logs, heartbeat,
cancellation acknowledgement, cancellation, interruption, disconnect,
reconnect, artifact readiness, completion, and failure. Every message binds
the immutable request identity, run ID, exact positive sequence, kind-specific
payload, protocol version, and its own content identity.

Payload schemas are closed. They reject absolute Windows or POSIX paths, URLs,
host or user fields, machine/process identifiers, and unexpected keys. Public
run evidence contains portable target, capability, recipe, version, content,
checkpoint, transfer, and stable failure facts. Private model paths, worker
roots, WSL distribution names, process state, and raw operational diagnostics
remain outside canonical records.

The final normalized artifact has exactly:

- `adapter.bin`, whose exact bytes are content-identified;
- `adapter_config.json`, which binds adapter structure, compatibility,
  runtime identity, payload format, and observed library versions; and
- `provenance.json`, which binds the producing run, frozen request, experiment,
  recipe resolution, dataset, and final training state.

Artifact integrity is rechecked before evaluation inference, focused or batch
local use, and export. Export retains the existing verified bundle contract and
does not assert deployment readiness.

## Capability and target truth

`probe()` reports public-safe accelerator backend, architecture and model
class, device count and memory, system memory, supported precision and
quantization modes, runtime capabilities, and library versions. These facts
are converted to the existing `HardwareCapabilityProfile`; the existing
preflight estimator and constraint checks decide readiness before launch.

The target class is immutable for a request. WSL and direct native execution
are selected explicitly and share the same backend contract. The runtime does
not retry on another target, rewrite a recipe, or hide a platform-driven
material change. A changed target or training configuration remains a derived
experiment under the accepted product contract.

## Verification strategy

Unit and contract tests cover closed message schemas, replay conflicts,
terminal races, resource ownership, staging corruption and partial data,
no-shell launch arguments, terminal-response reuse, callback failure cleanup,
library-backed service completion, cancellation, interruption, checkpoint
recovery, artifact reopen, evaluation inference, focused and single-item batch
local use, and verified export. The deterministic backend requires no network,
GPU, downloaded model, or private data.

The hardware test is explicitly opt-in. It probes the configured WSL ROCm
environment, runs at most two training steps against already-local model and
tokenizer sources, and executes evaluation inference. It skips with a precise
unmet-capability reason when opt-in or any local prerequisite is absent. The
default repository gate remains hardware-independent.

The library calls follow the supported local-loading and adapter APIs described
by the upstream [PEFT model documentation](https://huggingface.co/docs/peft/package_reference/peft_model),
[PEFT configuration guide](https://huggingface.co/docs/peft/en/tutorial/peft_model_config),
and [Accelerate `Accelerator` documentation](https://huggingface.co/docs/accelerate/main/package_reference/accelerator).
