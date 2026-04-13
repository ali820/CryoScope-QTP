#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN=".venv/bin/python"
else
  DEFAULT_PYTHON_BIN="python3"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
PYCACHE_PREFIX="${PYCACHE_PREFIX:-./tmp/pycache}"

env PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" "$PYTHON_BIN" -m py_compile \
  cryoscope_core.py \
  cryoscope_qtp_streamlit_prototype_v2.py \
  tests/test_core.py \
  scripts/smoke_check.py

env PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" "$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'
env PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" "$PYTHON_BIN" scripts/smoke_check.py
