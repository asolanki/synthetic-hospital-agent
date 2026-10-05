"""Compare complete fixed-dev reports, pairing patients and retaining failures.

    uv run python -m training.compare --reference BASE.json --candidate RL.json --name NAME
    uv run python -m training.compare --reference SFT-seed*.json --candidate RL-seed*.json --name NAME

Reports must come from the same unchanged paper-prompt evaluation protocol.
Several reports per arm are sampling replicates: each patient's reward is
averaged within an arm before pairing. This reads saved scores only; it never
evaluates or opens heldout patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from common import benchmark, config, metrics


def split_ids(split: str) -> set[int]:
    """Patients of a fixed split: our train/dev partition, or the paper's public split. Never heldout."""
    if split == "public":
        return {i.gt_id for i in benchmark.load_instances(("public",))}
    if split not in ("train", "dev"):
        raise ValueError(f"Unsupported split {split!r}; heldout is sealed")
    return set(json.loads(config.SPLIT_FILE.read_text())[split])


def _load(path: Path, split: str, ids: set[int]) -> tuple[dict, dict[int, dict]]:
    report = json.loads(path.read_text())
    rows = report["records"]
    records = {row["gt_id"]: row for row in rows}
    if (report["split"] != split or not report["complete_split"]
            or len(records) != len(rows) or set(records) != ids):
        raise ValueError(f"Expected each fixed {split} patient exactly once: {path}")
    if any(not math.isfinite(row["reward"]) for row in rows):
        raise ValueError(f"Nonfinite reward in {path}")
    return report, records


def load_arm(paths: list[Path], split: str, ids: set[int]) -> tuple[list[dict], dict[int, list[dict]]]:
    reports, rows = [], {gt_id: [] for gt_id in ids}
    for path in paths:
        report, records = _load(path, split, ids)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if any(seen["sha256"] == digest for seen in reports):
            raise ValueError(f"Duplicate report within one arm: {path}")
        reports.append({"path": str(path), "sha256": digest, "summary": report["summary"]})
        for gt_id, row in records.items():
            rows[gt_id].append(row)
    return reports, rows


def compare(reference_paths: list[Path], candidate_paths: list[Path], split: str = "dev") -> dict:
    ids = split_ids(split)
    reference, before = load_arm(reference_paths, split, ids)
    candidate, after = load_arm(candidate_paths, split, ids)
    pairs = []
    for gt_id in sorted(ids):
        reference_reward = statistics.mean(row["reward"] for row in before[gt_id])
        candidate_reward = statistics.mean(row["reward"] for row in after[gt_id])
        pairs.append({"gt_id": gt_id,
                      "reference_reward": reference_reward,
                      "candidate_reward": candidate_reward,
                      "delta": candidate_reward - reference_reward,
                      "reference_ok": [row["ok"] for row in before[gt_id]],
                      "candidate_ok": [row["ok"] for row in after[gt_id]]})
    mean_delta = lambda rows: statistics.mean(row["delta"] for row in rows)
    wins = sum(row["delta"] > 1e-12 for row in pairs)
    losses = sum(row["delta"] < -1e-12 for row in pairs)
    return {
        "split": split,
        "method": (f"Paired patient bootstrap, 1000 resamples, seed 0; all fixed {split} patients, failures "
                   "retained; replicate reports within an arm are averaged per patient before pairing"),
        "reference": reference,
        "candidate": candidate,
        "paired": {"n": len(pairs), "reference_replicates": len(reference),
                   "candidate_replicates": len(candidate), "mean_reward_difference": mean_delta(pairs),
                   "ci95": metrics.bootstrap_ci(pairs, mean_delta),
                   "wins": wins, "losses": losses, "ties": len(pairs) - wins - losses},
        "patients": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("dev", "train", "public"), default="dev")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.split)
    path = config.write_result(f"training/{args.name}", result)
    print(json.dumps(result["paired"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
