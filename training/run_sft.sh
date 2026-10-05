#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
exec "$ROOT/.venv-sft/bin/python" -m training.sft "$@"
