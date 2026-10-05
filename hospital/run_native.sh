#!/usr/bin/env bash
# Private service: access remotely through SSH forwarding, never a public listener.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
set -a
source "$ROOT/.env"
set +a
cd "$ROOT/external/synthetic_hospital"
exec "$ROOT/.venv-hospital/bin/uvicorn" epic_sim.app.main:app \
    --host 127.0.0.1 --port 58000 --workers 4
