# Temper ML Repository Instructions

## Public Repository Safety

This repository is public. Treat every staged byte and commit metadata field as
permanently publishable.

- Never commit credentials, tokens, keys, passwords, cookies, connection
  strings, secret values, or private URLs.
- Never commit non-public personal or operational identifiers, including direct
  email addresses, local usernames, home-directory paths, device names, serial
  numbers, IP or MAC addresses, account or organization IDs, run IDs, or
  private artifact identifiers and fingerprints.
- Never commit real private datasets, model weights, adapters, checkpoints,
  runtime stores, MLflow data, logs, generated review packets, or production
  experiment records.
- Never commit a digest, revision, filename, descriptor, provenance record, or
  retrieval URI derived from a private artifact.
- Fixture hashes are allowed only when their synthetic source bytes are
  committed and publicly reviewable.
- Never include internal thread names, delivery IDs, worktree locations,
  host/process identities, full environment dumps, signed URLs, or commands
  containing absolute paths.
- Use `<repo-root>`, synthetic identities, and small synthetic fixtures in
  documentation and tests.
- Store only secret names in examples. Use obvious inert placeholders.
- Do not force-add an ignored file without explicit maintainer approval.
- Inspect staged changes and commit metadata for public suitability before
  every commit and push.
- Before pushing, inspect staged binary content, commit messages, and every
  reachable commit's author and committer identity.
- Treat `.gitignore` as defense in depth, not as a privacy boundary.
- Use a repository-local GitHub noreply identity for commits.

Public architecture, public dependency names, generic algorithms, and
deliberately public fixture hashes are allowed.

## Subagents

- Do not use fast mode for subagents.

## Setup and Test Commands

- Set up development dependencies: `bash scripts/codex-cloud-setup.sh`
- Run repository maintenance checks: `bash scripts/codex-cloud-maintenance.sh`
- Run unit tests: `uv run pytest tests/unit`
- Show the fixture walkthrough command help: `bash scripts/temper-fixture-walkthrough.sh --help`
