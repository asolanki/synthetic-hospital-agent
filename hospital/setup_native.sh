#!/usr/bin/env bash
# Ubuntu 24.04 + Supervisor (e.g. a Vast CUDA container); no nested Docker.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PIN=911f34c4ac65a508543c4b3b90c373a0cd16534d
UPSTREAM="$ROOT/external/synthetic_hospital"

if [[ $(id -u) != 0 ]]; then
    echo 'Run as root inside the GPU container.' >&2
    exit 1
fi
command -v supervisorctl >/dev/null
command -v uv >/dev/null
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    postgresql-16 postgresql-client-16 redis-server libpq-dev build-essential

if [[ ! -d "$UPSTREAM/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/sparkcpark/synthetic_hospital "$UPSTREAM"
fi
git -C "$UPSTREAM" checkout --quiet "$PIN"
if [[ ! -x "$ROOT/.venv-hospital/bin/python" ]]; then
    uv venv --python 3.12 "$ROOT/.venv-hospital"
fi
uv pip install --python "$ROOT/.venv-hospital/bin/python" -r "$UPSTREAM/requirements-core.txt"
uv sync --locked --group verifiers --python 3.12

# Never copy a workstation .env to a marketplace host. Generate local-only secrets.
if [[ ! -e "$ROOT/.env" ]]; then
    umask 077
    python3 - <<'PY'
from pathlib import Path
import secrets
password = secrets.token_hex(24)
Path('.env').write_text('\n'.join([
    'SH_BASE_URL=http://127.0.0.1:58000',
    'OPENAI_API_KEY=local-only',
    f'EPIC_SIM_SCORER_TOKEN={secrets.token_hex(32)}',
    f'EPIC_SIM_JWT_SECRET={secrets.token_hex(32)}',
    f'PGPASSWORD={password}',
    f'EPIC_SIM_DATABASE_URL=postgresql+asyncpg://epic_sim:{password}@127.0.0.1:5432/epic_sim',
    f'EPIC_SIM_DATABASE_URL_SYNC=postgresql+psycopg://epic_sim:{password}@127.0.0.1:5432/epic_sim',
    f'DATABASE_URL=postgresql://epic_sim:{password}@127.0.0.1:5432/epic_sim',
    'EPIC_SIM_REDIS_URL=redis://127.0.0.1:6379/0',
]) + '\n')
PY
fi
set -a
source "$ROOT/.env"
set +a
export EPIC_SIM_SQLITE_SOURCE="$UPSTREAM/benchmark_v1.3.db"
export EPIC_SIM_ONTOLOGY_DIR="$UPSTREAM/data/ontology"
mkdir -p "$ROOT/runs/services"

cat > /etc/supervisor/conf.d/shrl-databases.conf <<EOF
[program:shrl-postgres]
command=/usr/lib/postgresql/16/bin/postgres -D /var/lib/postgresql/16/main -c config_file=/etc/postgresql/16/main/postgresql.conf -c listen_addresses=127.0.0.1
user=postgres
autostart=true
autorestart=unexpected
stdout_logfile=$ROOT/runs/services/postgres.log
redirect_stderr=true
stopasgroup=true
killasgroup=true

[program:shrl-redis]
command=/usr/bin/redis-server --bind 127.0.0.1 --port 6379 --save "" --appendonly no
autostart=true
autorestart=unexpected
stdout_logfile=$ROOT/runs/services/redis.log
redirect_stderr=true
stopasgroup=true
killasgroup=true
EOF
supervisorctl reread
supervisorctl update
for attempt in {1..60}; do
    if pg_isready -q -h 127.0.0.1 && redis-cli ping | grep -qx PONG; then break; fi
    sleep 1
done
pg_isready -h 127.0.0.1
redis-cli ping

# Pass the password through stdin rather than command arguments or log output.
python3 - <<'PY' | runuser -u postgres -- psql -v ON_ERROR_STOP=1 -q
import os
password = os.environ['PGPASSWORD'].replace("'", "''")
print("SELECT 'CREATE ROLE epic_sim LOGIN SUPERUSER' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname='epic_sim')\\gexec")
print(f"ALTER ROLE epic_sim PASSWORD '{password}';")
print("SELECT 'CREATE DATABASE epic_sim OWNER epic_sim' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname='epic_sim')\\gexec")
PY

cd "$UPSTREAM"
export PATH="$ROOT/.venv-hospital/bin:$PATH"
alembic -c epic_sim/alembic/alembic.ini upgrade head
COUNT=$(psql "$DATABASE_URL" -Atc 'SELECT count(*) FROM longitudinal_patients')
if [[ "$COUNT" == 0 ]]; then
    python -m epic_sim.migrate.sqlite_to_pg --source "$EPIC_SIM_SQLITE_SOURCE"
fi
python -m epic_sim.migrate.load_terminology --auto --download-icd10
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/setup_bash_readonly.sql

cat > /etc/supervisor/conf.d/shrl-hospital.conf <<EOF
[program:shrl-hospital]
command=/bin/bash $ROOT/hospital/run_native.sh
directory=$ROOT
autostart=true
autorestart=unexpected
stdout_logfile=$ROOT/runs/services/hospital.log
redirect_stderr=true
stopasgroup=true
killasgroup=true
stopwaitsecs=30
EOF
supervisorctl reread
supervisorctl update
for attempt in {1..120}; do
    if curl -fsS http://127.0.0.1:58000/health; then
        printf '\nHospital ready. Load credentials with: set -a; source .env; set +a\n'
        exit 0
    fi
    sleep 2
done
echo "Hospital did not become healthy; inspect $ROOT/runs/services/hospital.log" >&2
exit 1
