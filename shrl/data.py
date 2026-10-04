"""Harness-side read access to the released benchmark database (SQLite).

Used for building splits, hand-written audit strategies, and analysis. Nothing
here is ever shown to the policy. `visible_sections` reproduces what the /env
route lets the policy read (every section except assessment and plan).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

from shrl import config


def connect() -> sqlite3.Connection:
    if not config.BENCHMARK_DB.exists():
        raise FileNotFoundError(
            f"{config.BENCHMARK_DB} not found; run scripts/setup_env.sh to fetch Synthetic Hospital")
    conn = sqlite3.connect(f"file:{config.BENCHMARK_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class Instance:
    gt_id: int
    patient_id: int
    split: str
    ground_truth: dict
    profile: dict
    encounters: list[dict] = field(default_factory=list)   # ordered; each has "sections"

    @property
    def gt_diagnoses(self) -> list[dict]:
        """Scored reference entries, with acuity as the scorer reads it."""
        out = []
        for key in ("active_diagnoses", "chronic_conditions"):
            for d in self.ground_truth.get(key, []):
                if d.get("excluded_nondiagnostic"):
                    continue
                acuity = "chronic" if key == "chronic_conditions" else d.get("acuity", "unspecified")
                out.append({**d, "acuity": "acute" if acuity == "unspecified" else acuity,
                            "list": key})
        return out

    def visible_text(self) -> str:
        return "\n".join(s["section_text"] or "" for e in self.encounters for s in e["sections"])


def load_instances(splits: tuple[str, ...] = ("train",)) -> list[Instance]:
    """patient_diagnosis instances with profile and policy-visible chart sections."""
    conn = connect()
    marks = ",".join("?" * len(splits))
    rows = conn.execute(
        f"""SELECT g.gt_id, g.patient_id, g.split, g.ground_truth, p.profile
            FROM benchmark_ground_truth g JOIN longitudinal_patients p USING (patient_id)
            WHERE g.task = ? AND g.is_diagnostic AND g.split IN ({marks}) ORDER BY g.gt_id""",
        (config.TASK, *splits)).fetchall()
    instances = {r["patient_id"]: Instance(r["gt_id"], r["patient_id"], r["split"],
                                           json.loads(r["ground_truth"]), json.loads(r["profile"] or "{}"))
                 for r in rows}
    encounters: dict[int, dict] = {}
    for r in conn.execute(
            """SELECT encounter_id, patient_id, encounter_date, encounter_type, chief_complaint,
                      encounter_order FROM longitudinal_encounters ORDER BY patient_id, encounter_order"""):
        if r["patient_id"] in instances:
            enc = {**dict(r), "sections": []}
            encounters[r["encounter_id"]] = enc
            instances[r["patient_id"]].encounters.append(enc)
    hidden = ",".join("?" * len(config.OUTCOME_SECTIONS))
    for r in conn.execute(
            f"""SELECT id AS section_id, encounter_id, section_type, section_text, section_order
                FROM encounter_ehr_sections WHERE lower(section_type) NOT IN ({hidden})
                ORDER BY encounter_id, section_order""", config.OUTCOME_SECTIONS):
        if r["encounter_id"] in encounters:
            encounters[r["encounter_id"]]["sections"].append(dict(r))
    conn.close()
    return list(instances.values())


@lru_cache(maxsize=1)
def diagnosis_table() -> list[dict]:
    """The benchmark's knowledge-graph diagnoses (display name, ICD-10-CM code, acuity)."""
    conn = connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT diagnosis_id, display_name, icd10_code, acuity FROM diagnoses "
        "WHERE icd10_code IS NOT NULL AND display_name IS NOT NULL")]
    conn.close()
    return rows


@lru_cache(maxsize=1)
def name_to_icd() -> dict[str, str]:
    """Exact (case-insensitive) display name -> ICD-10-CM code."""
    out: dict[str, str] = {}
    for d in diagnosis_table():
        out.setdefault(d["display_name"].strip().lower(), d["icd10_code"])
    return out


def category(code: str) -> str:
    """3-character ICD-10 category, normalized the way the scorer does."""
    return (code or "").upper().replace(".", "").strip()[:3]
