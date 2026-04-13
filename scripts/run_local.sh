#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/streamlit" ]]; then
  exec .venv/bin/streamlit run cryoscope_qtp_streamlit_prototype_v2.py "$@"
fi

exec streamlit run cryoscope_qtp_streamlit_prototype_v2.py "$@"
