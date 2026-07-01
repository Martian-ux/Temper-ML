#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

"${PYTHON:-python}" scripts/temper-gate.py maintenance
