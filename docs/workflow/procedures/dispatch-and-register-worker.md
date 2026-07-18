# Start root implementation or an exceptional worker

1. Confirm the task is `MAINTAINER_AUTHORIZED`, the exact base/worktree is current, operational records are outside the repository, and the packet is not stale.
2. Validate the task key, exact owned paths, selected model/effort, open questions, decision references, acceptance criteria, and worker limits.
3. Acquire one normalized ownership lease and activate the root task as the routine implementation writer. Do not create a writer subagent or a final-verifier agent.
4. Compute the effective exceptional worker limit as the minimum of global, phase, class, packet, route, ownership, and maintainer limits. A second implementation writer is non-routine and requires every independence guard plus explicit maintainer approval; never create a third.
5. Before any exceptional writer activation, including a concurrent second root task, record a durable maintainer decision whose reference, actor, approval, task key, subject identity, exact base, owned paths, writer mode, reason, and authoritative provenance all bind the registered task. A caller-supplied boolean is not authorization.
6. For an approved exception only, pass that decision reference and write `SPAWN_REQUESTED` before one manual creation call. Bind the task, scope, paths, base, declared and selected route, observation availability, compliance, experiment label, and attempt.
7. On timeout, connection loss, partial response, unclear result, or a missing reference, record `SPAWN_UNKNOWN`, count it as active, and do not retry. Inspect active/recent workers and operational records before any maintainer replacement decision.
8. Coordinate an exceptional worker or the one cold reviewer through completion, finding, blocked, error, and maintainer-input events. Use the bounded heartbeat only to reconcile a suspected lost event. User-facing progress remains independent: meet the host-required update cadence without polling or waking the worker merely to produce an update.
9. Record runtime values only from supported telemetry. If unavailable, record `UNAVAILABLE` observation and `UNVERIFIED` compliance; never infer either from prompt text.
10. Repair accepted in-scope findings autonomously. Stop on the factual conditions in `AGENTS.md`, not on an arbitrary repair count.
11. Stop on base, route selection, ownership, scope, authorization, or safety drift and preserve the exact operational state for disposition.
