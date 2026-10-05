#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export OMP_NUM_THREADS=1
# Optional local base checkpoint, API model name, and unmerged rank-16 adapter.
if (( $# > 3 )); then
    echo "Usage: $0 [local-checkpoint [served-model-name [lora-adapter]]]" >&2
    exit 2
fi
MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
NAME="${2:-$MODEL}"
REVISION_ARGS=()
if [[ "$MODEL" == "Qwen/Qwen3-4B-Instruct-2507" ]]; then
    REVISION_ARGS=(--revision cdbee75f17c01a7cc42f958dc650907174af0554)
elif [[ ! -d "$MODEL" ]]; then
    echo "Local checkpoint directory does not exist: $MODEL" >&2
    exit 2
fi
LORA_ARGS=()
BASE_NAME="$NAME"
if [[ -n "${3:-}" ]]; then
    if [[ ! -f "$3/adapter_model.safetensors" || ! -f "$3/adapter_config.json" ]]; then
        echo "LoRA adapter files are missing: $3" >&2
        exit 2
    fi
    # Distinct names prevent requests for the adapter from resolving to the base.
    BASE_NAME="$NAME-base"
    LORA_ARGS=(--enable-lora --max-lora-rank 16 --lora-dtype bfloat16
               --lora-modules "$NAME=$3")
fi
exec "$ROOT/.venv-inference/bin/vllm" serve "$MODEL" \
    "${REVISION_ARGS[@]}" --served-model-name "$BASE_NAME" "${LORA_ARGS[@]}" \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 \
    --max-num-batched-tokens 8192 --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --generation-config vllm --seed 0
