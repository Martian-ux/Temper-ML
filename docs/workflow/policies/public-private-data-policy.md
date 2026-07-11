# Public/private data policy

## public_safe_categories

Sanitized product contracts, accepted architecture decisions, contributor and workflow rules, `AGENTS.md`, empty public templates without operational identifiers, review and verification policies, public procedures, public dependency names, generic algorithms, and synthetic fixtures whose source bytes are committed and reviewable.

## private_restricted_categories

Raw transcripts; local absolute paths, usernames, device/host/network/process identities; worker/thread/provider run IDs; model-routing receipts and usage logs; internal event logs; recovery bundles, patches, archives, and untracked manifests; confidential reviews; private datasets/artifacts/weights/checkpoints/experiments; names, paths, hashes, fingerprints, revisions, or retrieval descriptors derived from private artifacts; full environment dumps; operational notes; private checkpoints; and private current-state snapshots.

## safeguards

1. Every generated record declares classification and allowed locations.
2. Public projections use repository-relative paths or `<repo-root>` and synthetic identities.
3. The path guard rejects private operation roots, absolute paths, home-directory fragments, worktree locations, symlink escapes, and generated recovery/output directories from public candidates.
4. Content scans check credentials/secrets, personal/operational identifiers, signed/private URLs, private artifact descriptors/fingerprints, raw logs, and forbidden binary classes.
5. Fixture provenance proves every published digest derives from committed synthetic bytes.
6. Generated files may enter the repository only through the exact manual projection allowlist and destination guard. No automated projection is claimed until separately authorized tooling exists.
7. Before staging, inspect untracked and proposed bytes, file types, metadata, paths, and diffs. Never force-add ignored material without approval.
8. Before commit/push, inspect staged bytes, binaries, message, author/committer identity, and newly reachable commits.
9. Content changes invalidate a prior public-safety result; the result must be bound to the exact output identity.
10. On a suspected leak, do not reproduce it; report safe category/severity, stop publication, and route incident handling.

## phase1_projection_boundary

Before `MAINTAINER_AUTHORIZED`, prohibit public repository modification, implementation branch/worktree creation, worker launch, repository projection, full-gate execution, staging, commits, and integration. A private projection manifest may record source selections, omissions, and output identity, but it remains private and is never staged.

The initial public projection uses exact source sections and rule IDs from the approved projection allowlist; whole private policy, routing, registry, or state files are not promoted by default. Optional validator/projection tooling does not add a YAML dependency.

Operational IDs are stored only in restricted private fields and are stripped, not hashed, when a public projection is produced. Hashing a private identifier does not make it public-safe. Public decisions reference public-safe evidence or generic rationale; private evidence links stay in private indices.
