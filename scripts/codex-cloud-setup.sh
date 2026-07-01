#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to set up Temper ML development dependencies." >&2
  exit 127
fi

uv sync --dev
