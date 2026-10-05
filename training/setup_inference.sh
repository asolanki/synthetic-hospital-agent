#!/usr/bin/env bash
# CUDA 12.8 inference environment; separate from the locked Verifiers environment.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV="$ROOT/.venv-inference"
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python 3.12 "$VENV"
fi
uv pip sync --python "$VENV/bin/python" -r "$ROOT/training/inference-requirements.txt"
mkdir -p "$ROOT/runs/services"
uv pip freeze --python "$VENV/bin/python" > "$ROOT/runs/services/inference-requirements.txt"

cat > /etc/supervisor/conf.d/shrl-vllm.conf <<EOF
[program:shrl-vllm]
command=/bin/bash $ROOT/training/serve.sh
directory=$ROOT
autostart=true
autorestart=unexpected
stdout_logfile=$ROOT/runs/services/vllm.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=2
redirect_stderr=true
stopasgroup=true
killasgroup=true
stopwaitsecs=60
EOF
supervisorctl reread
supervisorctl update
# Startup includes the pinned model download, compilation, and CUDA graph capture.
for attempt in {1..600}; do
    if curl -fsS http://127.0.0.1:8000/health; then
        echo 'Inference ready on 127.0.0.1:8000.'
        exit 0
    fi
    sleep 3
done
echo "Inference did not become healthy; inspect $ROOT/runs/services/vllm.log" >&2
exit 1
