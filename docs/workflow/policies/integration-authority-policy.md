# Integration authority policy

## public_phase1_integration_rules

The public Phase 1 projection retains these rules: exactly one registered integrator owns one plan/worktree after a separate maintainer-only `INTEGRATION_AUTHORIZED` decision; one registered final-candidate verifier reuses current targeted verification and runs the final repository gate once after cold review but before that decision; only approved units and bounded preapproved conflicts may later be integrated; public safety is inspected; failures return to the narrow phase that owns the problem; and no branch/worktree deletion or cleanup is implied. Private plan IDs, local paths, recovery references, and provider identifiers are omitted.
