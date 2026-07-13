#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'usage: scripts/temper-fixture-walkthrough.sh <project-root>\n'
  exit 0
fi

if [[ $# -ne 1 ]]; then
  printf 'usage: scripts/temper-fixture-walkthrough.sh <project-root>\n' >&2
  exit 2
fi

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON:-python}" -m temper_ml.cli fixture-workflow "$1"
