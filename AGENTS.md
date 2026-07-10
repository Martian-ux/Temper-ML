# Temper ML Repository Instructions

## Authority and instruction precedence

Within the project workflow, apply this order:

1. Platform system and developer instructions.
2. Current explicit maintainer instructions.
3. Non-overridable public-repository safety constraints.
4. Accepted product and architecture decisions.
5. Approved workflow and integration policies.
6. Maintainer-authorized plans and task packets.
7. Private current-state snapshots and checkpoints.
8. Worker recommendations and implementation choices.

A task packet narrows work but cannot override a higher authority. Private snapshots and checkpoints record operational state; they are not public project authority and cannot supersede accepted decisions or repository rules. A stale packet fails validation when its decision, base, ownership, authorization, or repository state changes. Conversation history and worker summaries are never authority.

## Public-repository safety

Treat every staged byte and commit/push metadata field as permanently public. Never publish secrets; non-public personal or operational identifiers; private paths, URLs, IDs, artifacts, datasets, logs, reviews, recovery data, or private-derived names/hashes. Use synthetic public fixtures and `<repo-root>`. Never force-add ignored files without maintainer approval; `.gitignore` is defense in depth, not authority. Before staging, inspect proposed bytes, file types, metadata, paths, and diffs. Before commit/push, inspect staged bytes, binary content, message, author/committer identity, and reachable commit metadata. Stop on suspected exposure.

## Product-contract authority

Implement only referenced accepted product clauses. Raw transcripts are evidence; confirmed product contracts are authority. Open questions are not defaults. Product behavior changes require maintainer confirmation and an explicit superseding decision.

## Operating mode and authorization

Use the coordinator lifecycle `RECONCILE -> DELIBERATE -> DECIDE -> PLAN -> DISPATCH -> VERIFY -> INTEGRATE -> CLOSE`. Invalid transitions fail closed. The separate Phase 1 authorization lifecycle is:

`PROPOSED -> REVIEWED_WITH_CORRECTIONS -> MAINTAINER_AUTHORIZED -> IMPLEMENTATION_READY -> IMPLEMENTING -> VERIFIED -> INTEGRATION_AUTHORIZED -> INTEGRATED`.

Only the maintainer may grant `MAINTAINER_AUTHORIZED`. Cold review may recommend authorization but may not grant it. Before that state, prohibit public repository modification, implementation branch/worktree creation, worker launch, repository projection, full-gate execution, staging, commits, pilot execution, and integration.

Private proposal evidence, snapshots, task packets, pilot packets, and authorization records may be created or corrected before authorization is effective. Those records do not authorize implementation. Only a complete, identity-bound authorization record that passes readiness validation is effective.

## Task classification

Classify every task with `docs/workflow/policies/model-routing-policy.yaml` before work. A validated task packet must name class, objective, exact base, scope, decisions/evidence, acceptance/non-goals, verification/review, route, stop conditions, authorization state, classification, active-mission reference, and a recorded procedural mission-fit decision. A technically valid task outside the current priority remains deferred unless the maintainer changes the mission.

## Model routing

Prompt text never selects or proves a model. Record intended and user/API-selected model and reasoning effort. Record runtime-observed values when exposed; otherwise use `UNVERIFIED` and never claim verified compliance. Unknown user/API selection blocks dispatch. The Phase 1 public surface covers only routine administration, bounded mechanical work, normal implementation, cold technical review, and maintainer decision gates. Routine work uses the least expensive sufficient route, elevation requires explicit justification, and automatic review stacking is prohibited. Unexercised premium or diversity routes remain private.

## Delegation and worker limits

The effective worker limit is the minimum of the global ceiling, current-phase limit, task-class limit, packet limit, model-route limit, available non-conflicting ownership capacity, and explicit maintainer restriction. Fast mode is prohibited.

Default active implementation writers: one. Phase 1 hard ceiling: two. A third implementation writer is prohibited. A second requires disjoint paths, independent acceptance criteria, no uncommitted-output dependency, no shared blocking decision, known integration order, and explicit recorded approval. Unknown or ambiguous ownership reduces capacity. No recursive delegation. Worker launching remains manual. Never retry an ambiguous spawn first.

## Task and path ownership

Every writer needs one active normalized ownership lease. Read-only overlap is allowed; write overlap or uncertainty is a collision. Reviewers have no write lease and do not count as implementation writers, but remain within separate worker and usage budgets. Stop on scope, base, worktree, or ownership drift.

## Context and state authority

Use the latest valid private checkpoint and selective retrieval before summarization, but never treat a checkpoint as public project authority. Load only referenced decisions, evidence, source/tests, applicable instructions, reusable verification, and stop conditions. Token, word, file, and section counts are telemetry only and never independently force compaction, replacement, refusal, truncation, or task splitting.

Use semantic boundaries such as phase completion, material scope or state change, superseded assumptions, repository/thread mismatch, obsolete logs, mission drift, current/history confusion, durable-record conflict, or a complete restart packet becoming available. Preserve typed current state: objective/phase, verified repository state, accepted decisions, blockers, task/ownership, evidence, reusable verification, risks, stop conditions, and exact next action. Semantic compaction is procedural in Phase 1, not a claim of automatic validator enforcement.

## Verification-result reuse

Query reusable verification before running checks. Workers run targeted checks; reviewers reuse them and run finding-specific checks only; one registered final-candidate verifier runs the full repository gate once after cold review and final assembly, before the maintainer integration decision. Bind results to exact subject identity, command semantics, scope, environment/lock identity, side effects, and invalidation conditions. When an untracked or generated input is relevant, bind its stable content identity, role, and scope; otherwise the result is `NON_REUSABLE`.

## Review requirements and multiplicity

Multiple files alone do not require review. Protected boundaries require one cold Terra-high review. Other triggers require one independent review only when recorded by the matrix. Multiple triggers on one exact subject strengthen the rubric but do not create multiple reviewers. A second reviewer requires one concrete unresolved finding and a recorded reason that another perspective may resolve it. Do not automatically stack reviewers or model perspectives.

## Git and worktree safety

Do not commit directly to the canonical branch. A clean worktree is not a globally clean repository. Preserve dirty candidates and unintegrated branches as evidence. Do not create an implementation branch/worktree before `MAINTAINER_AUTHORIZED`; do not reset, clean, restore destructively, rewrite, delete branches, or remove worktrees without verified recovery, disposition, and approval.

## Integration authority

Exactly one registered integrator owns an integration plan and worktree only after a separate maintainer-only `INTEGRATION_AUTHORIZED` decision. Full-gate and public-safety evidence may recommend that decision but cannot grant it. The integrator may act only on the approved candidate and cannot rewrite decisions, add unrelated work, or delete recovery material.

## Handoff requirements

Every worker handoff names task/worker records, exact base/head or patch identity, changed paths, route, acceptance evidence, verification references, applied decisions, scope/safety statement, open findings, and integration guidance. No handoff means no completed worker and no integration.

## Stop and escalation conditions

Stop on unknown base, missing or conflicting decisions, blocking questions, unsafe context, route mismatch, ambiguous spawn, ownership/scope breach, stale evidence, public-safety concern, unapproved dependency, or destructive action without recovery and approval. Escalate one precise question with evidence and options; do not broaden the task.

## Maintainer decision gates

Maintainer approval is required for Phase 1 authorization, product decisions, public-policy exceptions, authority changes, risk acceptance, backward-incompatible or irreversible behavior, worker replacement after ambiguous or stalled state, and destructive cleanup. Present a compact delta/options packet, not a request to replay prior investigation.

## Canonical repository commands

Use `python scripts/temper-gate.py setup`, `maintenance`, `unit`, `fixture-help`, or `all` only within an authorized task. Query reusable verification first; one registered final-candidate verifier runs `all` on the final assembled candidate under the bounded run policy. Only the exact authorized pilot may change gate behavior, and the program stops before integration to main.
