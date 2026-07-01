#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/temper-fixture-walkthrough.sh --help

TML-001 provides this command target for the future deterministic fixture
walkthrough. The walkthrough execution path is intentionally deferred until the
fixture skeleton stage.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

usage >&2
exit 2
