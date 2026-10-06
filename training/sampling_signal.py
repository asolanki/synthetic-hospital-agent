"""Would RL have anything to learn from? Group composition of k sampled answers per patient.

    uv run python -m training.sampling_signal --split train --name NAME A-seed0.json ... A-seed3.json

The k reports are k samples of one policy on one fixed split (e.g. temperature 1, top-p 1, the RL
rollout controls). It reports the quantities measured on the native RL run
(`rl-sft-paper-v1-exploration.json`, `rl-sft-paper-v1-group-spread.json`). Reads saved reports only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics

from common import benchmark, config, metrics
from training.compare import load_arm, split_ids

SCOPE = ("k sampled reports of one policy on one fixed split. For each reference diagnosis: 0/k "
         "samples submitted its scorer match key (no group signal toward it), mixed (learnable), or "
         "k/k (solved). Equal-reward patient groups have zero GRPO advantage. Exposure counts other "
         "training patients carrying the same key.")


def exposure_bucket(others: int) -> str:
    return "0 others" if others == 0 else "1-4 others" if others <= 4 else "5+ others"


def signal(paths: list[Path], split: str) -> dict:
    ids = split_ids(split)
    reports, rows = load_arm(paths, split, ids)
    k = len(reports)
    if k < 2:
        raise ValueError("Need at least two sampled reports of the same policy")
    train_ids = split_ids("train")
    instances = {i.gt_id: i for i in benchmark.load_instances(("train",)) if i.gt_id in ids | train_ids}
    if split == "public":
        instances.update({i.gt_id: i for i in benchmark.load_instances(("public",))})
    carriers = Counter(key for gt_id in train_ids
                       for key in {metrics.match_key(d["icd10"]) for d in instances[gt_id].gt_diagnoses})
    cells: dict[tuple[str, str], Counter] = {}
    equal = 0
    stds = []
    for gt_id in sorted(ids):
        inst = instances[gt_id]
        rewards = [row["reward"] for row in rows[gt_id]]
        equal += max(rewards) - min(rewards) < 1e-12
        stds.append(statistics.pstdev(rewards))
        keys = [metrics.predicted_keys(row["submission"]) for row in rows[gt_id]]
        final_id = max(inst.encounters, key=lambda e: e["encounter_order"])["encounter_id"]
        for d in inst.gt_diagnoses:
            key = metrics.match_key(d["icd10"])
            hits = sum(key in pred for pred in keys)
            outcome = "0/k" if hits == 0 else "k/k" if hits == k else "mixed"
            visit = "final" if d.get("first_encounter_id") == final_id else "earlier"
            others = carriers[key] - (gt_id in train_ids)
            for cell in ((visit, "all"), (visit, exposure_bucket(others))):
                cells.setdefault(cell, Counter())[outcome] += 1
    composition = [{"visit": visit, "training_exposure": bucket, "diagnoses": sum(c.values()),
                    **{o: c[o] / sum(c.values()) for o in ("0/k", "mixed", "k/k")}}
                   for (visit, bucket), c in sorted(cells.items())]
    return {"scope": SCOPE, "split": split, "k": k, "reports": reports, "patients": len(ids),
            "mean_reward": statistics.fmean(r["summary"]["mean_reward"] for r in reports),
            "equal_reward_groups": equal / len(ids), "mean_within_group_std": statistics.fmean(stds),
            "composition": composition}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--split", choices=("train", "dev", "public"), required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = signal(args.reports, args.split)
    path = config.write_result(f"training/{args.name}", result)
    print(f"k={result['k']} patients={result['patients']} mean_reward={result['mean_reward']:.3f} "
          f"equal_reward_groups={result['equal_reward_groups']:.3f} within_std={result['mean_within_group_std']:.3f}")
    for row in result["composition"]:
        print(f"{row['visit']:>8} {row['training_exposure']:>11} n={row['diagnoses']:>5} "
              f"0/k={row['0/k']:.3f} mixed={row['mixed']:.3f} k/k={row['k/k']:.3f}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
