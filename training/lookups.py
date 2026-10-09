"""Code-book use in saved traces: lookups, loops, and an error budget for the submitted codes.

    uv run python -m training.lookups --traces runs/<eval>/.../traces.jsonl \
        --report results/raw/training/<name>.json --split public --name <name>-lookups

Each reference diagnosis is classified by two facts: whether its scorer match key was submitted
(hit) and whether that key appeared among the codes the episode's own `icd10_search` results
listed (shown). Hit and shown, hit from memory, missed though shown (a selection error), missed and
never shown (a retrieval or naming error). Diagnoses split by visit (earlier encounters vs the final
encounter) and, for "unseen", by whether no training patient carries the key. Submissions come
from the report built from the same traces (`training.report`). Never reads heldout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from common import benchmark, config, metrics

RESULTS_HEADER = "ICD-10-CM 2025 code book results for"
RESULT_CODE = re.compile(r"^\s*\d+\.\s+([A-Z][0-9][0-9A-Z.]*)\b", re.M)
KINDS = ("hit_shown", "hit_memory", "miss_shown", "miss_not_shown")


def _episode(trace: dict) -> dict:
    """Lookups, queries and listed match keys for one episode's trace."""
    queries, shown, listed_codes = [], set(), set()
    for node in trace["nodes"]:
        msg = node.get("message") or {}
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            if (fn.get("name") or call.get("name")) == "icd10_search":
                args = fn.get("arguments") if "arguments" in fn else call.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:  # a malformed call still counts as a lookup attempt
                        args = {}
                query = args.get("query", "") if isinstance(args, dict) else ""
                queries.append(" ".join(str(query).lower().split()))
        content = msg.get("content")
        if msg.get("role") == "tool" and isinstance(content, str) and content.startswith(RESULTS_HEADER):
            for code in RESULT_CODE.findall(content):
                listed_codes.add(code.upper())
                shown.add(metrics.match_key(code))
    repeats = Counter(queries)
    return {"lookups": len(queries), "distinct_queries": len(repeats),
            "max_repeats": max(repeats.values(), default=0), "shown": shown, "listed_codes": listed_codes,
            "stop": trace.get("stop_condition"),
            "input_tokens": trace.get("num_input_tokens") or 0, "output_tokens": trace.get("num_output_tokens") or 0}


def analyze(traces_path: Path, report_path: Path, split: str) -> dict:
    if split not in ("dev", "public"):
        raise ValueError("Lookup budgets use held-out patients: dev or public")
    episodes = {}
    for line in traces_path.read_text().splitlines():
        if line.strip():
            trace = json.loads(line)["traces"][0]
            episodes[trace["task"]["data"]["gt_id"]] = _episode(trace)
    report = json.loads(report_path.read_text())
    submissions = {row["gt_id"]: row.get("submission") for row in report["records"]}
    if set(episodes) != set(submissions):
        raise ValueError("Traces and report cover different patients")

    train_ids = set(json.loads(config.SPLIT_FILE.read_text())["train"])
    source = ("public",) if split == "public" else ("train",)
    instances = {i.gt_id: i for i in benchmark.load_instances(source) if i.gt_id in episodes}
    exposure = Counter(key for inst in benchmark.load_instances(("train",)) if inst.gt_id in train_ids
                       for key in {metrics.match_key(d["icd10"]) for d in inst.gt_diagnoses})

    budget = {cell: Counter() for cell in (("earlier", "all"), ("earlier", "unseen"), ("final", "all"),
                                           ("final", "unseen"))}
    submitted = Counter()
    for gt_id, ep in episodes.items():
        inst, submission = instances[gt_id], submissions[gt_id]
        final_id = max(inst.encounters, key=lambda e: e["encounter_order"])["encounter_id"]
        predicted = metrics.predicted_keys(submission)
        for d in inst.gt_diagnoses:
            key = metrics.match_key(d["icd10"])
            part = "final" if d.get("first_encounter_id") == final_id else "earlier"
            kind = ("hit_" if key in predicted else "miss_") + (
                "shown" if key in ep["shown"] else ("memory" if key in predicted else "not_shown"))
            budget[(part, "all")][kind] += 1
            if exposure[key] == 0:
                budget[(part, "unseen")][kind] += 1
        for k in ("active_diagnoses", "chronic_conditions"):
            for entry in ((submission or {}).get(k) or []):
                if isinstance(entry, dict) and entry.get("icd10"):
                    code = str(entry["icd10"]).upper().strip()
                    submitted["codes"] += 1
                    submitted["exact_code_listed"] += code in ep["listed_codes"]
                    submitted["match_key_listed"] += metrics.match_key(code) in ep["shown"]

    eps = list(episodes.values())
    lookups = [e["lookups"] for e in eps]
    return {
        "split": split, "traces": str(traces_path), "report": str(report_path), "episodes": len(eps),
        "scope": __doc__.split("\n\n")[2].replace("\n", " "),
        "lookups_per_patient": {"mean": statistics.fmean(lookups), "median": statistics.median(lookups),
                                "max": max(lookups), "patients_with_none": sum(n == 0 for n in lookups)},
        "repeated_queries": {"episodes_with_a_query_repeated_5_plus": sum(e["max_repeats"] >= 5 for e in eps),
                             "max_repeats": max(e["max_repeats"] for e in eps)},
        "stop_conditions": dict(Counter(e["stop"] for e in eps)),
        "submitted_codes": {"n": submitted["codes"],
                            "exact_code_in_own_results": submitted["exact_code_listed"] / submitted["codes"]
                            if submitted["codes"] else None,
                            "match_key_in_own_results": submitted["match_key_listed"] / submitted["codes"]
                            if submitted["codes"] else None},
        "error_budget": [{"visit": part, "training_patients_with_key": bucket, "diagnoses": sum(c.values()),
                          **{kind: c[kind] / sum(c.values()) if c else None for kind in KINDS}}
                         for (part, bucket), c in budget.items()],
        "tokens": {"input": sum(e["input_tokens"] for e in eps), "output": sum(e["output_tokens"] for e in eps)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "public"), default="public")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = analyze(args.traces, args.report, args.split)
    path = config.write_result(f"training/{args.name}", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("scope",)}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
