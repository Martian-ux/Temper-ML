# Dispatch and register worker

1. Confirm the task is `MAINTAINER_AUTHORIZED`, the exact base/worktree is current, operational records are outside the repository, and the packet is not stale.
2. Validate the task key, exact owned paths, selected model/effort, open questions, decision references, acceptance criteria, and worker limits.
3. Compute the effective worker limit as the minimum of global, phase, class, packet, route, ownership, and maintainer limits. Default to one writer; never create a third. A second requires every accepted independence guard.
4. Write an operational worker record in `SPAWN_REQUESTED` before one manual creation call. The record binds unique task key, scope, owned paths, base, intended and selected route, observed route or `UNVERIFIED`, and the attempt number.
5. On timeout, connection loss, partial response, unclear result, or a missing worker reference, record `SPAWN_UNKNOWN`, count it as active, and do not retry. Inspect available active/recent workers and known operational records, matching task key, scope, and ownership before any replacement decision.
6. After the maintainer returns the exact manual-launch worker reference, register it, confirm the selected route evidence, and continue automatically from the durable waiting state. A manual adapter is never described as fully automatic.
7. Record observed route fields. If runtime values are unavailable, use `UNVERIFIED`; never infer compliance from prompt text.
8. Stop on base, route, ownership, scope, authorization, or safety drift and preserve the exact operational state for disposition.
