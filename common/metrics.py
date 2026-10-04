"""Aggregate per-instance scorer results.

Two aggregates are reported, because they differ and both matter:

`mean_reward`
    Mean of the per-episode reward returned by the environment
    (`weighted_problem_list_f1_neutral` on a one-item batch). This is what an RL
    trainer optimizes. Missing or empty submissions count as 0.

`official_f1`
    The paper's batch aggregation (eval/scoring.py): the harmonic mean of the
    mean severity-weighted recall and the mean chart-neutral precision. Empty
    submissions are *excluded* from both means (scorer "tier C"), and instances
    whose chart-neutral precision is undefined are excluded from the precision
    mean. `official_f1_strict` is the same statistic with every excluded
    instance counted as recall 0 / precision 0, which is the fair number for
    comparing agents that differ in how often they fail to submit.

Each per-instance record is {"gt_id", "n_pred", "reward", "metrics"} where
`metrics` is the scorer's dict, or None when nothing was submitted.

`split_recall` adds one analysis column the official score does not have: recall on
diagnoses from earlier visits (written into later notes, so recall is transcription
plus ICD-10 coding) versus the final visit (never written down, must be inferred).
It is computed from the same submissions and changes nothing about the reward.
"""

from __future__ import annotations

import random
from typing import Callable


def _f1(recall: float, precision: float) -> float:
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _scored(rec: dict) -> bool:
    return bool(rec.get("metrics")) and rec["metrics"].get("n_scored", 0) == 1


def _neutral_precision(rec: dict) -> float | None:
    """Per-instance chart-neutral precision, or None where the batch scorer treats it as undefined.

    Undefined means every prediction was unmatched and chart-neutral, so the
    denominator is zero. The single-item scorer then falls back to raw precision
    (0), which would be wrong to average.
    """
    m = rec["metrics"]
    if rec["n_pred"] > 0 and m["n_neutral_predictions"] == rec["n_pred"] and m["problem_list_precision"] == 0:
        return None
    return m["problem_list_precision_neutral"]


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    scored = [r for r in records if _scored(r)]
    recall = [r["metrics"]["problem_list_recall"] for r in scored]
    w_recall = [r["metrics"]["weighted_problem_list_recall"] for r in scored]
    precision = [p for p in (_neutral_precision(r) for r in scored) if p is not None]
    confusion: dict[str, int] = {}
    for r in scored:
        for k, v in (r["metrics"].get("acuity_confusion_matrix") or {}).items():
            confusion[k] = confusion.get(k, 0) + v
    acuity_total = sum(confusion.values())
    acuity_correct = sum(v for k, v in confusion.items() if k.split(":")[0] == k.split(":")[1])
    n_unscored = n - len(scored)
    return {
        "n": n,
        "valid_submit_rate": len(scored) / n if n else 0.0,
        "mean_reward": _mean([r["reward"] for r in records]),
        "official_f1": _f1(_mean(w_recall), _mean(precision)),
        "official_f1_strict": _f1(sum(w_recall) / n if n else 0.0,
                                  sum(precision) / (len(precision) + n_unscored) if precision or n_unscored else 0.0),
        "weighted_recall": _mean(w_recall),
        "recall": _mean(recall),
        "precision_neutral": _mean(precision),
        "precision_raw": _mean([r["metrics"]["problem_list_precision"] for r in scored]),
        "acuity_accuracy": acuity_correct / acuity_total if acuity_total else None,
        "mean_predictions": _mean([r["n_pred"] for r in records]),
    }


def bootstrap_ci(records: list[dict], stat: Callable[[list[dict]], float],
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over instances (patients)."""
    if not records:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(records)
    values = sorted(stat([records[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return values[int(alpha / 2 * n_boot)], values[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def summarize(records: list[dict], n_boot: int = 1000) -> dict:
    out = aggregate(records)
    for key in ("mean_reward", "official_f1_strict"):
        lo, hi = bootstrap_ci(records, lambda rs, k=key: aggregate(rs)[k], n_boot=n_boot)
        out[f"{key}_ci95"] = [lo, hi]
    return out


def count_predictions(prediction) -> int:
    if not isinstance(prediction, dict):
        return 0
    n = 0
    for key in ("active_diagnoses", "chronic_conditions"):
        value = prediction.get(key)
        if isinstance(value, list):
            n += len(value)
    return n


# -- earlier-visit vs final-visit recall ---------------------------------------

def match_key(code: str) -> str:
    """The scorer's match rule: 3-character category, 5 characters for injury/poisoning (S, T)."""
    norm = (code or "").upper().replace(".", "").strip()
    return norm[:5] if norm[:1] in ("S", "T") else norm[:3]


def predicted_keys(submission) -> set[str]:
    if not isinstance(submission, dict):
        return set()
    return {match_key(d.get("icd10") or "") for k in ("active_diagnoses", "chronic_conditions")
            for d in (submission.get(k) or []) if isinstance(d, dict)}


def split_counts(inst, submission) -> dict[str, int]:
    """Hits and totals for one patient. `inst` is a common.benchmark.Instance."""
    final_id = max(inst.encounters, key=lambda e: e["encounter_order"])["encounter_id"]
    preds = predicted_keys(submission)
    out = {"earlier_hit": 0, "earlier_total": 0, "final_hit": 0, "final_total": 0}
    for d in inst.gt_diagnoses:
        part = "final" if d.get("first_encounter_id") == final_id else "earlier"
        out[f"{part}_total"] += 1
        out[f"{part}_hit"] += match_key(d["icd10"]) in preds
    return out


def split_recall(pairs) -> dict:
    """Micro-averaged recall over (instance, submission) pairs, unweighted."""
    totals = {"earlier_hit": 0, "earlier_total": 0, "final_hit": 0, "final_total": 0}
    for inst, submission in pairs:
        for k, v in split_counts(inst, submission).items():
            totals[k] += v
    return {
        "recall_earlier_visits": totals["earlier_hit"] / totals["earlier_total"] if totals["earlier_total"] else None,
        "recall_final_visit": totals["final_hit"] / totals["final_total"] if totals["final_total"] else None,
        "n_earlier": totals["earlier_total"], "n_final": totals["final_total"],
    }
