"""Paths and connection settings. Everything is overridable from the environment."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_URL = os.environ.get("SH_BASE_URL", "http://localhost:58000")
SCORER_TOKEN = os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production")

# The released SQLite database. Used harness-side only (splits, audit strategies,
# analysis); the policy never sees it.
BENCHMARK_DB = Path(os.environ.get(
    "SH_BENCHMARK_DB", ROOT / "external" / "synthetic_hospital" / "benchmark_v1.3.db"))

SPLIT_FILE = ROOT / "splits" / "patient_diagnosis.json"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

TASK = "patient_diagnosis"
SUBMIT_TOOL = "submit_diagnosis"
DEFAULT_BUDGET = 40

# Sections the environment strips from every observation (they state the answer).
OUTCOME_SECTIONS = ("assessment", "plan")
