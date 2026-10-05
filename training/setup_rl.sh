#!/usr/bin/env bash
# Ubuntu 24.04 / A100 rental: isolate CUDA 13 from the CUDA 12.8 evaluation stack.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PRIME="$ROOT/external/prime-rl"
REVISION=d9adc1e867db9f8e27438fdd65f99e8dc7a2448a
COMPAT="$ROOT/runs/services/cuda-compat"
DEB=cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb
mkdir -p "$ROOT/external" "$COMPAT"
if [[ ! -d "$PRIME/.git" ]]; then
    git clone --no-checkout https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME"
    git -C "$PRIME" checkout --detach "$REVISION"
fi
if [[ "$(git -C "$PRIME" rev-parse HEAD)" != "$REVISION" ]]; then
    echo "Existing Prime-RL checkout differs from the recorded revision; refusing to replace it" >&2
    exit 1
fi
git -C "$PRIME" -c url.https://github.com/.insteadOf=git@github.com: \
    submodule update --init --depth 1 -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
PATCH="$ROOT/training/prime-rl-shutdown.patch"
if ! git -C "$PRIME" apply --reverse --check "$PATCH" 2>/dev/null; then
    git -C "$PRIME" apply --check "$PATCH"
    git -C "$PRIME" apply "$PATCH"
fi
if [[ ! -f "$COMPAT/$DEB" ]]; then
    curl --fail --location --output "$COMPAT/$DEB" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/$DEB"
fi
printf '%s  %s\n' f7e29a545c1334bb5ca4b054213de9c9ceb70f8bf561aa214075020c4e0b60cf "$COMPAT/$DEB" | sha256sum --check
if [[ ! -e "$COMPAT/rootfs/usr/local/cuda-13.0/compat/libcuda.so.1" ]]; then
    dpkg-deb --extract "$COMPAT/$DEB" "$COMPAT/rootfs"
fi
export LD_LIBRARY_PATH="$COMPAT/rootfs/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PRIME"
uv sync --frozen --extra gpu --extra flash-attn --extra quack --extra kernels --extra dashboard --no-dev
uv pip install --python .venv/bin/python --no-deps -e "$ROOT/training/environment"
uv pip freeze --python .venv/bin/python > "$ROOT/runs/services/rl-requirements.txt"
uv run --no-sync python -c 'import prime_rl.trainer.model, synthetic_hospital_dx, verifiers; print("Prime-RL trainer and the hospital environment import successfully")'
