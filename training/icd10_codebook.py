"""Public ICD-10-CM code book: fetch it, and measure how far it can close the coding gap.

    uv run python -m training.icd10_codebook fetch
    uv run python -m training.icd10_codebook check --name codebook-check-dev \
        results/raw/training/qwen3-4b-sft-paper-dev.json ...

`check` (train/dev only) reports, for every dev reference diagnosis, the rank at which a search for
its exact reference name returns the right scorer match key. Then, for each model report, it
rescores the model's own answers after replacing every submitted code with the top search result
for the submitted name (`training.recode` does the same with the benchmark's private table).
Failed episodes stay at zero.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import statistics

import httpx

from common import benchmark, config, metrics
from common.simulator import Simulator
from synthetic_hospital_dx import icd10
from training.compare import split_ids

RANKS = (1, 3, 5, 8)


def fetch() -> Path:
    directory = icd10.codebook_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / icd10.ZIP_NAME
    if not path.exists():
        with httpx.stream("GET", icd10.CMS_URL, follow_redirects=True, timeout=300) as response:
            response.raise_for_status()
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != icd10.CMS_SHA256:
        path.unlink()
        raise ValueError(f"CMS release checksum mismatch: {digest}")
    book = icd10.load()
    print(json.dumps({"zip": str(path), "sha256": digest, "codes": len(book.titles),
                      "documents": len(book.documents)}))
    return path


def name_recall(book: icd10.CodeBook) -> dict:
    """Rank of the first correct match key when searching each dev reference diagnosis by name."""
    dev = split_ids("dev")
    rows = []
    for inst in benchmark.load_instances(("train",)):
        if inst.gt_id not in dev:
            continue
        final_id = max(inst.encounters, key=lambda e: e["encounter_order"])["encounter_id"]
        for d in inst.gt_diagnoses:
            hits = book.search(d["display_name"], limit=max(RANKS))
            keys = [metrics.match_key(h.code) for h in hits]
            target = metrics.match_key(d["icd10"])
            rows.append({"visit": "final" if d.get("first_encounter_id") == final_id else "earlier",
                         "rank": keys.index(target) + 1 if target in keys else None,
                         "exact_title": d["display_name"].strip().lower()
                         == book.titles.get(d["icd10"], "").strip().lower()})
    out = {}
    for visit in ("all", "earlier", "final"):
        sub = [r for r in rows if visit == "all" or r["visit"] == visit]
        out[visit] = {"diagnoses": len(sub),
                      "exact_official_title": sum(r["exact_title"] for r in sub) / len(sub),
                      **{f"recall_at_{k}": sum(r["rank"] is not None and r["rank"] <= k for r in sub) / len(sub)
                         for k in RANKS}}
    return out


def recode_top1(submission, book: icd10.CodeBook) -> tuple[dict | None, int]:
    if not isinstance(submission, dict):
        return submission, 0
    out, changed = copy.deepcopy(submission), 0
    for key in ("active_diagnoses", "chronic_conditions"):
        for entry in out.get(key) or []:
            if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
                continue
            hits = book.search(str(entry["name"]), limit=1)
            if hits and metrics.match_key(hits[0].code) != metrics.match_key(entry.get("icd10") or ""):
                entry["icd10"] = hits[0].code
                changed += 1
    return out, changed


async def rescore(report_path: Path, book: icd10.CodeBook) -> dict:
    report = json.loads(report_path.read_text())
    if report["split"] not in ("train", "dev") or not report["complete_split"]:
        raise ValueError(f"Expected a complete train/dev report: {report_path}")
    instances = {i.gt_id: i for i in benchmark.load_instances(("train",))}
    recoded = [(r, *recode_top1(r["submission"], book)) for r in report["records"]]
    async with Simulator() as sim:
        scored = iter(await sim.score_batch([(r["gt_id"], a) for r, a, _ in recoded if a is not None]))
    records = []
    for record, answer, _ in recoded:
        result = next(scored) if answer is not None else None
        records.append({"gt_id": record["gt_id"], "n_pred": metrics.count_predictions(answer),
                        "reward": result["reward"] if result else 0.0,
                        "metrics": result["metrics"] if result else None})
    summary = metrics.summarize(records)
    summary["visit_split"] = metrics.split_recall([(instances[r["gt_id"]], a) for r, a, _ in recoded])
    deltas = [{"d": new["reward"] - old["reward"]} for new, old in zip(records, report["records"])]
    mean = lambda rows: statistics.fmean(r["d"] for r in rows)
    return {"source": str(report_path), "source_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "entries_recoded": sum(c for _, _, c in recoded),
            "original_mean_reward": report["summary"]["mean_reward"],
            "recoded_mean_reward": summary["mean_reward"],
            "paired_difference": mean(deltas), "paired_ci95": metrics.bootstrap_ci(deltas, mean),
            "recoded_summary": {k: summary[k] for k in ("mean_reward", "mean_reward_ci95", "official_f1",
                                                        "official_f1_strict", "visit_split")},
            "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    check = sub.add_parser("check")
    check.add_argument("reports", type=Path, nargs="*")
    check.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
        return
    book = icd10.load()
    result = {"source": icd10.CMS_URL, "sha256": icd10.CMS_SHA256, "name_search_recall_dev": name_recall(book),
              "top1_recode": {}}
    for path in args.reports:
        result["top1_recode"][path.stem] = asyncio.run(rescore(path, book))
    out = config.write_result(f"training/{args.name}", result)
    print(json.dumps(result["name_search_recall_dev"], indent=1))
    for stem, r in result["top1_recode"].items():
        print(stem, json.dumps({k: r[k] for k in ("entries_recoded", "original_mean_reward", "recoded_mean_reward",
                                                  "paired_difference", "paired_ci95")}))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
