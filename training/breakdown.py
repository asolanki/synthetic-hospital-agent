"""Where did a checkpoint change? Slice paired fixed-split reports (dev or public).

    uv run python -m training.breakdown --reference SFT.json --candidate RL.json --name NAME

Diagnosis recall is unweighted and uses the scorer's match key. It is split by
visit (earlier encounters vs the final encounter) and by how many of the 680
training patients carry the same key ("unseen" = none; this is fine-tuning
exposure, not pretraining exposure). Patient rewards are paired within the
encounter-count buckets used to stratify the split. Behaviour compares the
reports' own summaries plus answer changes. Replicate reports within an arm are
averaged; answer-change counts use each arm's first report. Never reads heldout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from common import benchmark, config, metrics, splits
from training.compare import load_arm, split_ids

SCOPE = ("Exploratory slices of fixed-split reports. Unweighted match-key recall by visit and by how many "
         "training patients carry the key (fine-tuning exposure, not pretraining); paired reward by "
         "encounter-count bucket; behaviour from report summaries. Replicates within an arm are averaged; "
         "answer changes use each arm's first report.")

BEHAVIOUR = ("mean_actions", "mean_nudges", "mean_output_tokens", "mean_predictions",
             "precision_neutral", "weighted_recall", "acuity_accuracy", "valid_submit_rate")


def familiarity(n_train_patients: int) -> str:
    return "unseen" if n_train_patients == 0 else "1" if n_train_patients == 1 else (
        "2-4" if n_train_patients <= 4 else "5+")


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _name(text) -> str:
    return " ".join(str(text or "").lower().replace(",", " ").split()).rstrip(".")


def gold_outcome(diagnosis: dict, submission) -> str:
    """hit: match key submitted; named_miscoded: reference name submitted under another code; not_named."""
    entries = [e for k in ("active_diagnoses", "chronic_conditions")
               for e in ((submission or {}).get(k) or []) if isinstance(e, dict)]
    if metrics.match_key(diagnosis["icd10"]) in metrics.predicted_keys(submission):
        return "hit"
    if _name(diagnosis.get("display_name")) in {_name(e.get("name")) for e in entries}:
        return "named_miscoded"
    return "not_named"


def breakdown(reference_paths: list[Path], candidate_paths: list[Path], split: str = "dev") -> dict:
    if split not in ("dev", "public"):
        raise ValueError("Breakdowns compare held-out patients: dev or public")
    ids = split_ids(split)
    train_ids = set(json.loads(config.SPLIT_FILE.read_text())["train"])
    reference, before = load_arm(reference_paths, split, ids)
    candidate, after = load_arm(candidate_paths, split, ids)
    source = ("public",) if split == "public" else ("train",)
    instances = {i.gt_id: i for i in benchmark.load_instances(source) if i.gt_id in ids}
    instances.update({i.gt_id: i for i in benchmark.load_instances(("train",)) if i.gt_id in train_ids})
    exposure = Counter(key for gt_id in train_ids
                       for key in {metrics.match_key(d["icd10"]) for d in instances[gt_id].gt_diagnoses})

    hits = defaultdict(lambda: {"n": 0, "reference": 0.0, "candidate": 0.0})
    outcomes = {arm: defaultdict(Counter) for arm in ("reference", "candidate")}
    patients = defaultdict(list)
    answers = Counter()
    for gt_id in sorted(ids):
        inst = instances[gt_id]
        final_id = max(inst.encounters, key=lambda e: e["encounter_order"])["encounter_id"]
        ref_keys = [metrics.predicted_keys(row["submission"]) for row in before[gt_id]]
        cand_keys = [metrics.predicted_keys(row["submission"]) for row in after[gt_id]]
        gold = set()
        for d in inst.gt_diagnoses:
            key = metrics.match_key(d["icd10"])
            gold.add(key)
            part = "final" if d.get("first_encounter_id") == final_id else "earlier"
            for arm, rows in (("reference", before[gt_id]), ("candidate", after[gt_id])):
                for row in rows:
                    outcome = gold_outcome(d, row["submission"])
                    for cell in ((part, "all"), (part, familiarity(exposure[key]))):
                        outcomes[arm][cell][outcome] += 1 / len(rows)
            for cell in ((part, familiarity(exposure[key])), (part, "all"), ("all", familiarity(exposure[key])),
                         ("all", "all")):
                hits[cell]["n"] += 1
                hits[cell]["reference"] += _mean(key in keys for keys in ref_keys)
                hits[cell]["candidate"] += _mean(key in keys for keys in cand_keys)
        reference_reward = _mean(row["reward"] for row in before[gt_id])
        candidate_reward = _mean(row["reward"] for row in after[gt_id])
        patients[splits.encounter_bucket(len(inst.encounters))].append(
            {"gt_id": gt_id, "delta": candidate_reward - reference_reward,
             "reference_reward": reference_reward, "candidate_reward": candidate_reward})
        first_ref, first_cand = ref_keys[0], cand_keys[0]
        answers["patients"] += 1
        answers["changed_answers"] += first_ref != first_cand
        answers["correct_keys_gained"] += len((first_cand - first_ref) & gold)
        answers["correct_keys_lost"] += len((first_ref - first_cand) & gold)
        answers["wrong_keys_added"] += len((first_cand - first_ref) - gold)
        answers["wrong_keys_removed"] += len((first_ref - first_cand) - gold)

    recall = [{"visit": part, "training_patients_with_key": bucket, "diagnoses": cell["n"],
               "reference_recall": cell["reference"] / cell["n"],
               "candidate_recall": cell["candidate"] / cell["n"],
               "difference": (cell["candidate"] - cell["reference"]) / cell["n"]}
              for (part, bucket), cell in sorted(hits.items())]
    mean_delta = lambda rows: statistics.fmean(row["delta"] for row in rows)
    complexity = [{"encounters": bucket, "patients": len(rows),
                   "reference_mean_reward": _mean(r["reference_reward"] for r in rows),
                   "candidate_mean_reward": _mean(r["candidate_reward"] for r in rows),
                   "mean_reward_difference": mean_delta(rows),
                   "ci95": metrics.bootstrap_ci(rows, mean_delta)}
                  for bucket, rows in sorted(patients.items())]
    behaviour = {key: {"reference": _mean(r["summary"].get(key) for r in reference
                                          if r["summary"].get(key) is not None),
                       "candidate": _mean(r["summary"].get(key) for r in candidate
                                          if r["summary"].get(key) is not None)}
                 for key in BEHAVIOUR}
    gold_outcomes = [{"arm": arm, "visit": part, "training_patients_with_key": bucket,
                      "diagnoses": round(sum(counts.values())),
                      **{kind: counts[kind] / sum(counts.values()) for kind in ("hit", "named_miscoded", "not_named")}}
                     for arm, cells in outcomes.items() for (part, bucket), counts in sorted(cells.items())]
    return {"scope": SCOPE,
            "reference": reference, "candidate": candidate,
            "recall_by_visit_and_training_exposure": recall,
            "gold_outcomes_by_visit": gold_outcomes,
            "reward_by_encounter_count": complexity,
            "split": split, "behaviour": behaviour, "answer_changes": dict(answers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("dev", "public"), default="dev")
    args = parser.parse_args()
    result = breakdown(args.reference, args.candidate, args.split)
    path = config.write_result(f"training/{args.name}", result)
    for row in result["recall_by_visit_and_training_exposure"]:
        print(f"{row['visit']:>8} {row['training_patients_with_key']:>6} n={row['diagnoses']:>3} "
              f"{row['reference_recall']:.3f} -> {row['candidate_recall']:.3f} ({row['difference']:+.3f})")
    for row in result["reward_by_encounter_count"]:
        lo, hi = row["ci95"]
        print(f"encounters {row['encounters']:>4} n={row['patients']:>3} "
              f"{row['mean_reward_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(json.dumps({"gold_outcomes_by_visit": result["gold_outcomes_by_visit"],
                      "behaviour": result["behaviour"], "answer_changes": result["answer_changes"]}, indent=1))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
