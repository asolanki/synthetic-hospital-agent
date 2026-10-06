"""Add public code-book lookups to existing oracle-copy demonstrations (augmented harness).

    uv run python -m training.sft_data_icd10 --source oracle-copy-v1 --name oracle-copy-icd10-v1 \
        --tools-from data/sft/oracle-copy-icd10-v1/tools_source_trace.jsonl

Each demonstration keeps its recorded chart observations and reference submission. Before the
submission, one assistant turn calls `icd10_search` once per submitted diagnosis name, followed by
the tool's real output from the frozen public ICD-10-CM index. The submission is unchanged, so
when a search does not surface the reference category the demonstration still submits the
reference code (label-informed, like the source data). Train patients only.
"""
from __future__ import annotations

import argparse
import json

from common import config, metrics
from synthetic_hospital_dx import icd10

SEARCH_TOOL = "icd10_search"


def augment(row: dict, book: icd10.CodeBook) -> tuple[dict, int, int]:
    """Return the row with a search turn inserted, entries searched, and reference keys surfaced."""
    messages = row["messages"]
    submit_at = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    submit_call = messages[submit_at]["tool_calls"][0]["function"]
    if submit_call["name"] != "submit_diagnosis":
        raise ValueError(f"gt_id={row['gt_id']}: last assistant turn is not the submission")
    answer = json.loads(submit_call["arguments"])
    entries = [e for key in ("active_diagnoses", "chronic_conditions") for e in answer[key]]
    calls, results, surfaced = [], [], 0
    for n, entry in enumerate(entries):
        call_id = f"search_{n}"
        hits = book.search(entry["name"])
        surfaced += metrics.match_key(entry["icd10"]) in {metrics.match_key(h.code) for h in hits}
        calls.append({"id": call_id, "type": "function",
                      "function": {"name": SEARCH_TOOL, "arguments": json.dumps({"query": entry["name"]})}})
        results.append({"role": "tool", "tool_call_id": call_id, "content": icd10.render(entry["name"], hits)})
    search_turn = {"role": "assistant", "content": "", "tool_calls": calls}
    out = {**row, "messages": [*messages[:submit_at], search_turn, *results, *messages[submit_at:]]}
    return out, len(entries), surfaced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tools-from", required=True)
    args = parser.parse_args()
    episode = json.loads(open(args.tools_from).readline())
    tools = [t if "function" in t else {"type": "function", "function": {
        k: t[k] for k in ("name", "description", "parameters") if k in t}} for t in episode["traces"][0]["tools"]]
    if SEARCH_TOOL not in {t["function"]["name"] for t in tools}:
        raise ValueError(f"{args.tools_from} has no {SEARCH_TOOL} tool; use a trace from the augmented harness")
    book = icd10.load()
    source = config.SFT_DIR / args.source / "train.jsonl"
    out_dir = config.SFT_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    searched = surfaced = rows = 0
    with (out_dir / "train.jsonl").open("w") as f:
        for line in source.read_text().splitlines():
            row, n, s = augment(json.loads(line), book)
            row["tools"] = json.dumps(tools)
            f.write(json.dumps(row) + "\n")
            rows, searched, surfaced = rows + 1, searched + n, surfaced + s
    meta = {"name": args.name, "source": args.source, "split": "train", "n": rows,
            "policy": "view_encounters -> view_encounter_detail(latest) -> icd10_search(each name) -> submit(reference)",
            "searches": searched, "reference_key_in_results": surfaced / searched,
            "codebook": {"url": icd10.CMS_URL, "sha256": icd10.CMS_SHA256, "index": icd10.INDEX_NAME},
            "tools_from": args.tools_from}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
