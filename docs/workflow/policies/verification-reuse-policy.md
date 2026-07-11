# Verification-result reuse policy

## public_phase1_rules

The Phase 1 public projection retains only these durable rules: bind results to exact subject identity, command semantics, scope, relevant environment and dependency identity, side effects, and invalidation; query reusable results before running checks; have reviewers run finding-specific checks only; and have one registered final-candidate verifier run the full repository gate once after cold review and final assembly, before the maintainer integration decision. Numeric packet or context sizes are telemetry only and never trigger workflow action by themselves. Relevant untracked or generated inputs require stable content identities, roles, and scopes or the result is `NON_REUSABLE`.
