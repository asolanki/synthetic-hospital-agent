"""A first look at the benchmark database: the numbers in the blog's "Looking at the data" section.

Run from the repo root: uv run python -m audit.explore_data
Reads train and public labels only; heldout labels are never loaded.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

from common.config import BENCHMARK_DB, TASK


def category(code: str) -> str:
    """The scorer's matching key: 3 characters, or 5 for chapter S and T codes."""
    c = code.replace(".", "").upper()
    return c[:5] if c[:1] in "ST" else c[:3]


def main() -> None:
    con = sqlite3.connect(f"file:{BENCHMARK_DB}?mode=ro", uri=True)
    patients = pd.read_sql("SELECT * FROM longitudinal_patients", con)
    visits = pd.read_sql("SELECT * FROM longitudinal_encounters", con)
    sections = pd.read_sql("SELECT encounter_id, section_type FROM encounter_ehr_sections", con)
    key = pd.read_sql(
        "SELECT patient_id, split, ground_truth FROM benchmark_ground_truth "
        "WHERE task = ? AND split != 'heldout'", con, params=(TASK,))

    print(f"{len(patients):,} patients, {len(visits):,} visits, {len(sections):,} note sections")
    print("\nColumn types (JSON columns arrive as plain strings):")
    print(patients.dtypes.to_string())

    # Two kinds of patient.
    n_visits = visits.groupby("patient_id").size()
    print("\nVisits per patient:")
    print(n_visits.value_counts().sort_index().to_string())
    print(f"2-3 visits: {(n_visits <= 3).sum()}, 8 visits: {(n_visits == 8).sum()}")
    chars = visits.groupby("patient_id").note_text.apply(lambda s: s.str.len().sum())
    print(f"Sections per note (median): {sections.groupby('encounter_id').size().median():.0f}")
    print(f"Characters per note (median): {visits.note_text.str.len().median():,.0f}")
    print(f"Characters per 8-visit chart (median): {chars[n_visits == 8].median():,.0f}; "
          f"longest chart: {chars.max():,}")

    # The answer key is short.
    key["dx"] = key.ground_truth.map(json.loads).map(
        lambda g: g["active_diagnoses"] + g["chronic_conditions"])
    scored = key.dx.map(lambda ds: [d for d in ds if not d.get("excluded_nondiagnostic")])
    n_dx = scored.map(len)
    per_visit = n_dx / key.patient_id.map(n_visits)
    print(f"\nReference diagnoses per patient (train + public, scored): median {n_dx.median():.0f}, "
          f"mean {n_dx.mean():.2f}, per visit {per_visit.mean():.2f}")
    for split in ("train", "public"):
        print(f"  {split}: mean {n_dx[key.split == split].mean():.2f}")
    chronic = patients.profile.map(lambda p: len(json.loads(p).get("chronic_conditions", [])))
    print(f"Chronic conditions in each profile before the first visit (mean): {chronic.mean():.2f}")

    # The codes have a long tail.
    rows = [(pid, category(d["icd10"])) for pid, ds in zip(key.patient_id, scored) for d in ds]
    cats = pd.DataFrame(rows, columns=["patient_id", "category"])
    patients_per_cat = cats.groupby("category").patient_id.nunique()
    print(f"\nScorer categories: {len(patients_per_cat)}; "
          f"in only one patient: {(patients_per_cat == 1).mean():.0%}")
    print("Largest ICD-10 chapters (share of diagnoses):")
    print(cats.category.str[0].value_counts(normalize=True).head(5).round(3).to_string())

    # The generator leaves fingerprints.
    uncoded = [d["display_name"] for ds in key.dx for d in ds if not d.get("icd10")]
    flagged = [d for ds in key.dx for d in ds if d.get("excluded_nondiagnostic")]
    print(f"\nAnswer-key entries with no ICD-10 code: {len(uncoded)}, "
          f"all {sum(not d.get('icd10') for d in flagged)} flagged excluded_nondiagnostic; "
          f"the scorer skips all {len(flagged)} flagged entries")
    for name in uncoded[:8]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
