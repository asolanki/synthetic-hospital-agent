"""Counterfactual: rescore a model's own answers with the benchmark's codes for the names it chose.

    uv run python -m training.recode results/raw/training/REPORT.json --name NAME

Each submitted entry whose name exactly matches (case-insensitive) a diagnosis in the benchmark's
own table gets that table's ICD-10 code; other entries keep the model's code. The model still
decides which diagnoses to submit, so this isolates ICD coding from chart reading. It is not a
deployable agent: the table is the benchmark's knowledge graph. Failed episodes stay at zero.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path

from common import benchmark, config, metrics
from common.simulator import Simulator
from training.compare import split_ids

SCOPE = ("Submitted entries whose name exactly matches (case-insensitive) a benchmark diagnosis get that "
         "diagnosis's ICD-10 code; other entries keep the model's code. Isolates coding from which diagnoses "
         "the model chose. Not a deployable agent: the table is the benchmark's knowledge graph.")


def recode(submission: dict | None, table: dict[str, str]) -> tuple[dict | None, int, int]:
    """Return the recoded submission, entries considered, and entries whose code changed."""
    if not isinstance(submission, dict):
        return submission, 0, 0
    out, entries, changed = copy.deepcopy(submission), 0, 0
    for key in ("active_diagnoses", "chronic_conditions"):
        for entry in out.get(key) or []:
            if not isinstance(entry, dict):
                continue
            entries += 1
            code = table.get(str(entry.get("name") or "").strip().lower())
            if code and metrics.match_key(code) != metrics.match_key(entry.get("icd10") or ""):
                entry["icd10"] = code
                changed += 1
    return out, entries, changed


async def rescore(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    split = report["split"]
    if not report["complete_split"] or {r["gt_id"] for r in report["records"]} != split_ids(split):
        raise ValueError(f"Expected a complete fixed-split report: {report_path}")
    table = benchmark.name_to_icd()
    source = ("public",) if split == "public" else ("train",)
    instances = {i.gt_id: i for i in benchmark.load_instances(source)}
    recoded, entries, changed = [], 0, 0
    for record in report["records"]:
        answer, n, c = recode(record["submission"], table)
        recoded.append((record, answer))
        entries, changed = entries + n, changed + c
    async with Simulator() as sim:
        scored = iter(await sim.score_batch([(r["gt_id"], a) for r, a in recoded if a is not None]))
    records = []
    for record, answer in recoded:
        # Episodes that never produced an answer remain zero, as in the original report.
        result = next(scored) if answer is not None else None
        records.append({"gt_id": record["gt_id"], "n_pred": metrics.count_predictions(answer),
                        "reward": result["reward"] if result else 0.0,
                        "metrics": result["metrics"] if result else None})
    summary = metrics.summarize(records)
    summary["visit_split"] = metrics.split_recall([(instances[r["gt_id"]], a) for r, a in recoded])
    return {"source": str(report_path), "source_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "split": split, "scope": SCOPE, "entries": entries, "entries_recoded": changed,
            "original_summary": {k: report["summary"][k] for k in
                                 ("mean_reward", "mean_reward_ci95", "official_f1", "official_f1_strict",
                                  "visit_split")},
            "recoded_summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = asyncio.run(rescore(args.report))
    path = config.write_result(f"training/{args.name}", result)
    keys = ("mean_reward", "mean_reward_ci95", "official_f1", "official_f1_strict", "visit_split")
    print(json.dumps({"entries": result["entries"], "entries_recoded": result["entries_recoded"],
                      "original": result["original_summary"],
                      "recoded": {k: result["recoded_summary"][k] for k in keys}}, indent=1))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
