#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ROOT/runs/services/cuda-compat/rootfs/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT/external/prime-rl"
# One inference engine needs no separate Rust load-balancing router.
exec uv run --no-sync inference @ "$ROOT/training/configs/rl_inference.toml" "$@"
