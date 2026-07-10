# Worktree and branch policy

## public_phase1_worktree_rules

The public Phase 1 projection retains these rules: a clean worktree is not a globally clean repository; preserved dirty evidence is not a worker sandbox; implementation branch/worktree creation requires a current `MAINTAINER_AUTHORIZED` record and exact base; review is read-only; integration has one registered authority; retirement is non-destructive; and destructive cleanup requires recovery proof, disposition, no active owner, and explicit maintainer approval. Private aliases, local paths, recovery locations, and preserved candidate identifiers are removed from the projection.
