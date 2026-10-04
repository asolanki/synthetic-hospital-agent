"""Paths and simulator connection settings shared by audit/, baselines/ and training/.

Everything is overridable from the environment. The Verifiers environment in
training/environment/ deliberately does not import this: prime-rl loads it as a
standalone package, so it carries its own copy of the two simulator settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The simulator started by hospital/setup.sh.
BASE_URL = os.environ.get("SH_BASE_URL", "http://localhost:58000")
SCORER_TOKEN = os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production")

# The released SQLite database. Read by our analysis code only (splits, audit
# strategies, metrics); the policy never sees it.
BENCHMARK_DB = Path(os.environ.get(
    "SH_BENCHMARK_DB", ROOT / "external" / "synthetic_hospital" / "benchmark_v1.3.db"))

SPLIT_FILE = ROOT / "data" / "splits" / "patient_diagnosis.json"
SFT_DIR = ROOT / "data" / "sft"
RESULTS_DIR = ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
RUNS_DIR = ROOT / "runs"          # full trajectories, gitignored

TASK = "patient_diagnosis"
SUBMIT_TOOL = "submit_diagnosis"
DEFAULT_BUDGET = 40

# Sections the environment strips from every observation (they state the answer).
OUTCOME_SECTIONS = ("assessment", "plan")


def load_dotenv() -> None:
    """Read KEY=value lines from the repo's .env (API keys) without overriding the shell."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def write_result(relpath: str, payload: dict) -> Path:
    """Write a raw result as results/raw/<relpath>.json and return the path."""
    path = RAW_RESULTS_DIR / f"{relpath}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path
