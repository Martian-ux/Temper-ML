# Model route selection and matched experiments

## Select and verify a route

Route declarations, selection, observation, compliance, and experiment labels are separate facts.

1. Record the declared route and exact model/effort required by the task class.
2. Select them with an executable host mechanism, never with prompt wording:
   - desktop host: use the model and reasoning controls before task creation;
   - CLI: `codex exec --model <model-id> --config 'model_reasoning_effort="<effort>"' --json "<task>"`;
   - CLI profile: put `model` and `model_reasoning_effort` in the selected profile and run `codex exec --profile <profile> --json "<task>"`;
   - API host: set the model and effort in the task-creation request.
3. Record the selection mechanism as a structured object. Its `kind` must be `host_controls`, `cli_flags`, `cli_profile`, or `api_parameters`, and its model/effort selectors must match the executable surface documented in `model-routing-policy.yaml`. Host controls also name the control surface; profiles name the selected profile. Free-form strings, prompt instructions, and unknown selectors block implementation or reviewer launch.
4. Read runtime values from supported host task metadata when available. For CLI trials, configure a private OpenTelemetry exporter and read the `codex.conversation_starts` event, which includes the model and reasoning settings. `codex exec --json` is useful for lifecycle and usage metrics but does not by itself prove the runtime route.
5. Set observation availability to `OBSERVED` only when that telemetry exposes both values. Otherwise set it to `UNAVAILABLE`, keep observed values `UNVERIFIED`, and set declared-route compliance to `UNVERIFIED`.
6. When observation is available, set compliance to `PASS` for an exact match and `FAIL` for a mismatch. A mismatch blocks the run from being treated as route-compliant.

Codex configuration and telemetry receipts are operationally private. Never stage profiles, host metadata, OpenTelemetry endpoints, tokens, raw events, task IDs, or conversation IDs. The supported selection and telemetry surfaces are documented in the [Codex configuration guide](https://learn.chatgpt.com/docs/config-file/config-basic) and [monitoring guidance](https://learn.chatgpt.com/docs/agent-approvals-security#monitoring-and-telemetry).

## Ultra trial shape

The trial has exactly six matched pairs and twelve runs: one control and one experimental run per pair. The pair mix is two fixed-snapshot cold reviews, two bounded implementation or repair tasks, one architecture or invariant-design task, and one mechanical negative control.

Before either run begins, freeze one public-safe task packet containing the exact base, candidate or input identity, objective, allowed paths/tools, fixtures, acceptance criteria, non-goals, stop conditions, and scoring rubric. Randomize A/B execution order with a recorded seed. For implementation pairs, create two isolated worktrees at the same base and give each run the same frozen task; neither run may read the other worktree, branch, transcript, result, or telemetry. Reviews use the same immutable snapshot and independent read-only contexts.

If the tasks, bases, inputs, acceptance criteria, or worktree isolation differ, label the comparison `OBSERVATIONAL`. Observational results may inform a future hypothesis but are excluded from causal claims and default-route decisions.

## Normalized score

Adjudication is blind to route labels. Replace route names with deterministic A/B aliases using the predeclared seed. One recorded adjudicator applies the same ledger to both runs; a second model perspective is not added automatically.

After both runs, form one adjudication ledger from the predeclared acceptance/invariant items plus every independently reproduced real defect found in either output. Give each item a fixed severity weight: P1 = 5, P2 = 2, P3 or acceptance item = 1. Record pass/fail evidence for each run. A claimed finding counts only when a deterministic reproduction, accepted invariant, or maintainer ruling validates it; otherwise it is a false positive.

Compute these 0-100 components, clamping every result to that range:

Define `effective_tokens = max(0, input_tokens - cached_input_tokens) + output_tokens`; report cached-input and reasoning-output tokens separately but do not add reasoning twice when it is already included in output tokens.

- `quality = 100 * credited_ledger_weight / total_ledger_weight - min(20, 5 * false_positives)`. If the ledger has zero weight, quality is 100 only when both runs satisfy all acceptance criteria; otherwise it is 0.
- `autonomy = 100 - 20 * avoidable_blocking_clarifications - 10 * avoidable_tool_or_test_retries - 25 * incomplete_outcome`.
- For each efficiency metric -- effective tokens (50%), elapsed seconds (30%), agent sessions (10%), and redundant full-gate runs (10%) -- score a run as `100 * pair_min / run_value`. When both values are zero, both score 100; when only the run value is zero, it scores 100. The weighted mean is `efficiency`.
- `handoff = 10 * passed_items` for a predeclared ten-item binary checklist covering identity, scope, decisions, validation, findings, safety, route truthfulness, experiment label, reproducibility, and exact next action.
- `total = 0.50 * quality + 0.20 * autonomy + 0.20 * efficiency + 0.10 * handoff`.

Store raw measures, formulas, aliases, seed, ledger, evidence references, component scores, and total scores so another evaluator can reproduce the result. The adjudicator resolves disagreements only from reproducible evidence or an explicit maintainer ruling and records the resolution before aliases are revealed.

Register a completed comparison with `temper-workflow validate-route-trial` before using it as route-default evidence. The executable validator requires exactly six complete pairs; the required task mix; one bound `CONTROL` and one bound `EXPERIMENTAL` run per pair; unique run and isolated context identities; matching frozen-task, task-class, and isolation references; one shared ledger identity and total weight per pair; raw token, elapsed, session, redundant-gate, ledger-weight, false-positive, autonomy, and handoff measures; ledger evidence; pair-aware zero-weight quality; reproducible quality, autonomy, efficiency, handoff, and total scores; and reproducible aggregate deltas. A comparison registered as `OBSERVATIONAL` must set route-default eligibility to false. Passing validation makes a matched result eligible evidence only; it does not authorize a default change.

Use Ultra as a default for a task class only after the six-pair trial shows a repeatable material quality benefit without an escaped P1 regression. If aggregate quality is equivalent, retain the lower-effort route. The aggregate report must show per-pair deltas, mean/median total delta, quality delta, and every P1/P2 difference; a pooled total alone is insufficient.

Treat totals within one point as a tie and choose the lower-effort route. Any override requires an explicit maintainer risk decision citing the per-pair evidence.
