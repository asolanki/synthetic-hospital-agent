#!/usr/bin/env bash
# Keep trainer dependencies separate from both inference and the Verifiers lock.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV="$ROOT/.venv-sft"
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python 3.12 "$VENV"
fi
uv pip sync --python "$VENV/bin/python" -r "$ROOT/training/sft-requirements.txt"
mkdir -p "$ROOT/runs/services"
uv pip freeze --python "$VENV/bin/python" > "$ROOT/runs/services/sft-requirements.txt"
