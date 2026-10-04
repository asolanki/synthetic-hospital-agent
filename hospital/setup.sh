#!/bin/sh
# Fetch Synthetic Hospital at the pinned commit (into external/, gitignored) and start
# the simulator on :58000. Run from anywhere: hospital/setup.sh
set -eu
cd "$(dirname "$0")/.."

PIN=911f34c4ac65a508543c4b3b90c373a0cd16534d
DEST=external/synthetic_hospital

if [ ! -d "$DEST/.git" ]; then
    mkdir -p external
    git clone https://github.com/sparkcpark/synthetic_hospital "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$PIN"

docker compose -f hospital/compose.yaml up -d --build
echo "waiting for the simulator (first boot loads the database and downloads ICD-10-CM)..."
until curl -sf "http://localhost:${SH_PORT:-58000}/health" >/dev/null; do sleep 3; done
echo "simulator ready on http://localhost:${SH_PORT:-58000}"
