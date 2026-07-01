#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to run Temper ML maintenance checks." >&2
  exit 127
fi

uv run python -m compileall -q src
uv run pytest tests/unit
