#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  "${PYTHON:-python}" scripts/temper-gate.py fixture-help
  exit 0
fi

"${PYTHON:-python}" scripts/temper-gate.py fixture-help >&2
exit 2
