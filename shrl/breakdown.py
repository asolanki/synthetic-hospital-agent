"""Recall split by where a reference diagnosis sits in the patient's timeline.

The official score blends two different skills. Reference diagnoses established
before the final encounter are carried into later notes' "Active Problem List"
by name, so recalling them is transcription plus ICD-10 coding. Diagnoses
established at the final encounter are never carried forward and have to be
inferred from findings.

This is an analysis column only: it is computed from the same submissions the
official scorer sees and changes nothing about the environment or the reward.
Matching follows the scorer's rule (3-character category; 5 characters for
injury and poisoning codes), unweighted.
"""

from __future__ import annotations

import json
from pathlib import Path

from shrl import data


def _key(code: str) -> str:
    norm = (code or "").upper().replace(".", "").strip()
    return norm[:5] if norm[:1] in ("S", "T") else norm[:3]


def predicted_keys(submission) -> set[str]:
    if not isinstance(submission, dict):
        return set()
    return {_key(d.get("icd10") or "") for k in ("active_diagnoses", "chronic_conditions")
            for d in (submission.get(k) or []) if isinstance(d, dict)}


def split_counts(inst: data.Instance, submission) -> dict[str, int]:
    final_id = max(inst.encounters, key=lambda e: e["encounter_order"])["encounter_id"]
    preds = predicted_keys(submission)
    out = {"earlier_hit": 0, "earlier_total": 0, "final_hit": 0, "final_total": 0}
    for d in inst.gt_diagnoses:
        part = "final" if d.get("first_encounter_id") == final_id else "earlier"
        out[f"{part}_total"] += 1
        out[f"{part}_hit"] += _key(d["icd10"]) in preds
    return out


def split_recall(pairs: list[tuple[data.Instance, object]]) -> dict:
    """Micro-averaged recall over (instance, submission) pairs."""
    totals = {"earlier_hit": 0, "earlier_total": 0, "final_hit": 0, "final_total": 0}
    for inst, submission in pairs:
        for k, v in split_counts(inst, submission).items():
            totals[k] += v
    return {
        "recall_earlier_visits": totals["earlier_hit"] / totals["earlier_total"] if totals["earlier_total"] else None,
        "recall_final_visit": totals["final_hit"] / totals["final_total"] if totals["final_total"] else None,
        "n_earlier": totals["earlier_total"], "n_final": totals["final_total"],
    }


def for_run(traj_path: Path, instances: dict[int, data.Instance]) -> dict:
    latest: dict[int, dict] = {}
    for line in traj_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["gt_id"]] = rec
    return split_recall([(instances[r["gt_id"]], r["submission"]) for r in latest.values()
                         if r["stop_reason"] != "llm_error" and r["gt_id"] in instances])
