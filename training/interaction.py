"""Patient-paired interaction (difference in differences) of four complete reports on one split.

    uv run python -m training.interaction --split public --name NAME \
        --small-without SFT8.json --small-with TOOL_SFT.json \
        --large-without LUNA.json --large-with LUNA_BOOK.json

Per patient, I = (small_with - small_without) - (large_with - large_without): how much more the
intervention (here the code book) is worth to the small model than to the large one. Equivalently,
I = (large_without - small_without) - (large_with - small_with): how much the scale gap shrinks.
Replicate reports within an arm are averaged per patient, failures are retained, and the percentile
bootstrap resamples patients sorted by gt_id with 1,000 resamples and seed 0, exactly as
`training.compare` does, so the effects here match its pairwise contrasts. Never reads heldout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from common import config, metrics
from training.compare import load_arm, split_ids

ARMS = ("small_without", "small_with", "large_without", "large_with")
EFFECTS = {
    "small_effect": ("small_with", "small_without"),
    "large_effect": ("large_with", "large_without"),
    "gap_without": ("large_without", "small_without"),
    "gap_with": ("large_with", "small_with"),
}


def interaction(paths: dict[str, list[Path]], split: str) -> dict:
    ids = split_ids(split)
    reports, rows = {}, {}
    for arm in ARMS:
        reports[arm], rows[arm] = load_arm(paths[arm], split, ids)
    patients = []
    for gt_id in sorted(ids):
        reward = {arm: statistics.mean(row["reward"] for row in rows[arm][gt_id]) for arm in ARMS}
        row = {"gt_id": gt_id, **reward}
        for name, (a, b) in EFFECTS.items():
            row[name] = reward[a] - reward[b]
        row["interaction"] = row["small_effect"] - row["large_effect"]
        patients.append(row)

    def summary(key: str) -> dict:
        stat = lambda rs: statistics.fmean(r[key] for r in rs)
        positive = sum(r[key] > 1e-12 for r in patients)
        negative = sum(r[key] < -1e-12 for r in patients)
        return {"mean": stat(patients), "ci95": metrics.bootstrap_ci(patients, stat),
                "positive": positive, "negative": negative, "zero": len(patients) - positive - negative}

    return {
        "split": split,
        "method": ("Per-patient interaction I = (small_with - small_without) - (large_with - large_without) "
                   "= gap_without - gap_with, on mean per-patient official reward. Paired patient bootstrap, "
                   "1000 resamples, seed 0, patients sorted by gt_id (as training.compare); failures retained; "
                   "replicates within an arm averaged per patient."),
        "arms": {arm: reports[arm] for arm in ARMS},
        "n": len(patients),
        "effects": {name: summary(name) for name in (*EFFECTS, "interaction")},
        "patients": patients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}", dest=arm, type=Path, nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("dev", "train", "public"), default="public")
    args = parser.parse_args()
    result = interaction({arm: getattr(args, arm) for arm in ARMS}, args.split)
    path = config.write_result(f"training/{args.name}", result)
    print(json.dumps(result["effects"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
