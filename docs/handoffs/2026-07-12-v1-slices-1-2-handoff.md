# Temper ML v1 Slices 1 and 2 Handoff

**Status:** Slice 1 and Slice 2 implementation complete; draft pull-request
candidate pending cross-platform CI.

## Completed Slice 1 foundation

- Typed records are stored as complete canonical envelopes at their verified
  record identities under the project-local immutable store.
- Immutable writes use fsynced temporary regular files and a no-overwrite
  commit. Reads bind the opened file to its stable filesystem identity.
- Verification checks canonical bytes, registered schema and projection
  versions, typed payload invariants, filename/type binding, declared typed
  reference closure, embedded record closure, event chains, persisted bundle
  manifests, safe paths, and derived-state rebuildability.
- Event streams retain predecessor hashes and idempotency behavior. Payloads
  are deeply immutable, and record wire shapes are rejected until a future
  typed lifecycle-event schema can own their reference semantics.
- Canonical admission rejects secret-like fields and values, operational
  identifiers, unsafe paths, private URIs, network identifiers, control text,
  and other non-public material without echoing it in errors.
- Public dumps use a separate default-deny projection described by
  `schemas/public/public-dump-v1.schema.json`. Unclassified record payloads,
  identities, stream names, event types, and raw events are omitted.
- The CLI provides read-only `status`, `verify`, `dump`, and exact local
  `manifest` inspection with canonical UTF-8 JSON and stable error codes.
- Legacy registry, run, and artifact layout surfaces fail closed rather than
  being silently excluded from a successful typed-store verification.

## Slice 2 compatibility proof

The complete graph test constructs every registered Slice 2 record type,
including both experiment revisions, the exact manifest diff, and their
derivation. It writes the graph in reverse dependency order, reconstructs it
through Slice 1, verifies every pinned dependency, and checks that the public
projection contains no content identity or private canary.

The Slice 2 contract suites also cover incompatible comparison, merge, resume,
deployment, and runtime-target combinations; deterministic recipe resolution;
artifact availability; local use and export evidence; and exact experiment
derivation semantics.

## Reproducible fixture and recovery proof

`fixtures/projects/minimal` contains only committed synthetic bytes. The record
and event identities are reproducible from their canonical JSON, and the
rendering-contract identity is reproduced from the committed source file.
Canonical fixture JSON is pinned to LF on every platform, and only the exact
reviewed runtime-shaped files are unignored.

The fixture and temporary-project tests verify all four CLI commands, simulate
an interrupted immutable write, corrupt and remove derived state, rebuild it,
and prove that canonical record and event bytes remain unchanged. Cleanup of a
canonical record is rejected.

## Verification and boundaries

Publication uses the repository gate on the exact candidate:

```text
python scripts/temper-gate.py --bootstrap-uv temp all
```

The default gate requires no private data, accelerator, network service, or
external provider. Ubuntu and Windows CI must both pass before the pull request
leaves draft.

This work does not add project-creation services, dataset behavior, a runtime,
training, evaluation, UI, provider adapters, or hardware dependencies. Those
remain later slices in the adopted v1 execution roadmap. The next bounded work
is Slice 3: deterministic project, recipe, hardware, and experiment services
over these verified contracts.
