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
4. Read runtime values only from the supported `host_task_metadata` source when the host exposes them. For CLI trials, configure a private OpenTelemetry exporter and use only the `codex.conversation_starts` source, whose event includes the model and reasoning settings. Any other source name is not runtime-route evidence. `codex exec --json` is useful for lifecycle and usage metrics but does not by itself prove the runtime route.
5. Set observation availability to `OBSERVED` only when that telemetry exposes both values. Otherwise set it to `UNAVAILABLE`, keep observed values `UNVERIFIED`, and set declared-route compliance to `UNVERIFIED`.
6. When observation is available, set compliance to `PASS` for an exact match and `FAIL` for a mismatch. A mismatch blocks the run from being treated as route-compliant.

An exact-task maintainer route exception may authorize a selected-route mismatch
without rewriting the declared route or its compliance result. The executable
record must bind the maintainer approval to the task key, exact base, immutable
subject, complete owned-path set, selected model and effort, and the same
authority reference as the task authorization. It must include a non-empty
reason and state both `public_policy_change: false` and `precedent: false`.
Changing any binding invalidates the exception. Exceptions apply only to the
bound root task, are forbidden on experiment records, never qualify as matched
trial evidence, do not authorize a reviewer or additional writer, and do not
change the default route.

Codex configuration and telemetry receipts are operationally private. Never stage profiles, host metadata, OpenTelemetry endpoints, tokens, raw events, task IDs, or conversation IDs. The supported selection and telemetry surfaces are documented in the [Codex configuration guide](https://learn.chatgpt.com/docs/config-file/config-basic) and [monitoring guidance](https://learn.chatgpt.com/docs/agent-approvals-security#monitoring-and-telemetry).

## Ultra trial shape

The trial has exactly six matched pairs and twelve runs: one control and one experimental run per pair. The pair mix is two fixed-snapshot cold reviews, two bounded implementation or repair tasks, one architecture or invariant-design task, and one mechanical negative control.

The executable task-mix mapping is fixed: `fixed_snapshot_cold_review` uses `cold_technical_review`, `bounded_implementation_or_repair` uses `normal_implementation`, `architecture_or_invariant_design` uses `protected_boundary_implementation`, and `mechanical_negative_control` uses `mechanical_change`. Both runs in a pair must use the mapped class. A `sol-high` control is permitted only for non-mechanical pairs; the mechanical negative control must use its `terra-medium` default. A label/class or control-route mismatch cannot qualify as matched-trial evidence.

Before either run begins, freeze one public-safe task packet containing the exact base, candidate or input identity, objective, allowed paths/tools, fixtures, acceptance criteria, non-goals, stop conditions, and scoring rubric. Randomize A/B execution order with a recorded seed. For implementation pairs, create two isolated worktrees at the same base and give each run the same frozen task; neither run may read the other worktree, branch, transcript, result, or telemetry. Reviews use the same immutable snapshot and independent read-only contexts.

If the tasks, bases, inputs, acceptance criteria, or worktree isolation differ, label the comparison `OBSERVATIONAL`. Observational results may inform a future hypothesis but are excluded from causal claims and default-route decisions.

Every run registered in a `MATCHED_TRIAL` must also have `OBSERVED` runtime model and effort from one of the two supported telemetry sources, exact agreement with its declared model and effort, and `PASS` declared-route compliance. Telemetry unavailability, an unsupported source, or a mismatch prevents matched-trial eligibility; preserve the comparison as `OBSERVATIONAL` instead.

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

Register a completed comparison with `temper-workflow validate-route-trial` before using it as route-default evidence. The executable validator requires exactly six complete pairs; the required task mix and mapped task class; one bound `CONTROL` and one bound `EXPERIMENTAL` run per pair; supported observed telemetry and passing declared-route compliance for every run; globally unique run and context identities across all twelve runs; matching frozen-task, task-class, and isolation references; one shared ledger identity and total weight per pair; raw token, elapsed, session, redundant-gate, ledger-weight, false-positive, autonomy, and handoff measures; ledger evidence; pair-aware zero-weight quality; reproducible quality, autonomy, efficiency, handoff, and total scores; and reproducible aggregate deltas. A comparison registered as `OBSERVATIONAL` must set route-default eligibility to false. Passing validation makes a matched result eligible evidence only; it does not authorize a default change.

For every pair, derive `total_delta` and `quality_delta` as experimental minus control. Derive the result in this order:

- If the absolute mean total delta is at most one point, the outcome is `TIE`.
- Otherwise, the outcome is `MATERIAL_QUALITY_BENEFIT` only when mean total delta, median total delta, and mean quality delta are each greater than one point and at least four of six pairs have both total and quality deltas greater than one point.
- Every other outcome is `NO_MATERIAL_BENEFIT`.
- The recommendation is `ADOPT_EXPERIMENTAL` only for a material quality benefit with no escaped P1. Every tie, non-material result, or escaped P1 derives `KEEP_CONTROL`.

The validator recomputes the three aggregate deltas, outcome, and recommendation from the twelve run records and rejects caller-supplied values that differ. The aggregate report must show per-pair deltas, mean/median total delta, mean quality delta, and every P1/P2 difference; a pooled total alone is insufficient. The one-point tie boundary is inclusive and retains the lower-effort route. Any override requires an explicit maintainer risk decision citing the per-pair evidence.
