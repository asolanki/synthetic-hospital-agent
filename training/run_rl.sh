#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
set -a
source "$ROOT/.env"
set +a
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$ROOT/runs/services/cuda-compat/rootfs/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/external/prime-rl"
exec uv run --no-sync rl @ "$ROOT/training/configs/rl_diagnosis.toml" "$@"
