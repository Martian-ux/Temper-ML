# Slice 9 retention, cleanup, and reproduction architecture

## Scope and authority

This decision implements the accepted Slice 9 clauses in the v1 execution
roadmap and architecture design. It does not authorize canonical-record
deletion, automatic garbage collection, arbitrary paths, deployment behavior,
Slice 10 loops, or merge to `main`.

## Storage boundary

The canonical `.temper` store is retained in full and is never a cleanup
target. Cleanup can inspect and remove only verified regular files below the
fixed project-local `.temper-fixture-output` heavy-byte root. The service does
not accept an arbitrary cleanup root or filesystem path from a caller.

Inventory is a non-canonical snapshot. Each entry binds a portable logical key,
byte class, byte count, content identity, affected canonical subjects, impact
categories, and a private stable file-object observation. Physical groups are
derived from file-object equality, but device and inode values never appear in
views, canonical records, events, or public fixtures. Hard links therefore
count once as physical bytes, and a plan claims freed bytes only when every
known link is selected and no reference outside the inventory remains.

Full retention is the default. Nothing is selected implicitly. Runtime control
files, active staging, unknown byte classes, links/reparse points, non-regular
files, and every canonical surface fail closed.

## Cleanup protocol

Cleanup has three explicit stages:

1. `inventory -> plan` freezes the exact inventory identity and selected entry
   identities. The plan reports logical bytes selected, physical bytes that can
   actually be freed, retained and deleted classes, shared-reference effects,
   affected subjects, and separate warnings for resumability, inspectability,
   final-artifact availability, cache convenience, and debugging evidence.
2. `plan -> execute` requires explicit confirmation and a plan-scoped unique
   execution identity, rebuilds the inventory, rejects a stale or changed
   snapshot, and claims one nonblocking project cleanup lease before persisting
   the complete portable deletion intent. The lease serializes distinct
   execution identities and restart reconciliation so their event and
   availability histories cannot fork. Before any unlink the executor fences
   affected artifacts as unavailable and affected checkpoints as non-resumable.
   It then re-verifies every selected regular file and content identity, unlinks
   only those exact entries, and stops on the first failure.
3. `restart -> reconcile` reads an incomplete durable intent, combines its
   per-object deletion intents with the fixed-root filesystem state, settles the
   conservative availability fences, and materializes the missing completed,
   partial, or failed receipt. Reusing the same execution identity is
   idempotent; recreating identical bytes requires a new execution identity and
   therefore produces distinct evidence.

The executor appends an immutable cleanup lifecycle before and during removal.
It then writes superseding artifact-availability observations, checkpoint
removal events, and one immutable cleanup receipt. If a post-delete evidence
write fails, the pre-delete fence remains the current conservative observation
until reconciliation completes; absent bytes are never advertised as available
or resumable. Removing any required bundle member makes that artifact
unavailable. Canonical manifests, hashes, run events, metrics, recommendations,
lineage, and prior receipts remain untouched.

## Replay protocol

Replay plans remain non-canonical and content identified. A strict plan keeps
the original experiment manifest byte-for-byte unchanged. Execution recomputes
preflight from the launch request and rejects any experiment, resolution,
requirements, target, profile, estimate, or dataset mismatch before creating a
new run.

When strict replay is blocked, adaptation is explicit. Existing experiment
derivation creates a new experiment, an exact RFC 6901 manifest diff, and an
`adapted_reproduction` lineage record. The replay executor accepts that derived
experiment only when the adapted preflight is ready and records lifecycle
evidence that the result is adapted, never exact.

## Operator surface

The CLI exposes inventory, cleanup preview, confirmed cleanup execution, and
replay-plan views as stable JSON. A preview plan ID must be supplied again for a
separate cleanup execution invocation; changed bytes produce a different plan
and fail closed.

The local dashboard adds one storage-and-replay workspace to the existing calm,
dense operator shell. Its visual thesis is a storage ledger with consequence-
first actions: inventory is primary, cleanup impact appears before confirmation,
and replay mode plus exact manifest differences occupy the evidence detail
plane. The UI calls application services only and contains no cleanup or replay
policy.

## Verification invariants

- Synthetic isolated roots prove path confinement, link rejection, hard-link
  accounting, stale-plan rejection, serialized competing execution,
  partial-failure evidence, and restart-safe availability.
- Canonical-evidence adversarial coverage proves `.temper` survives cleanup.
- Strict replay produces a new run for the unchanged manifest; adapted replay
  produces a new run only for the persisted derived experiment and exact diff.
- Local-use, recovery, CLI, UI, schema, redaction, and complete repository-gate
  coverage remain green on Windows and Ubuntu.
