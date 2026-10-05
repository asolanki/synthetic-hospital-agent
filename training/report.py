"""Rescore saved Verifiers episodes with the official scorer and shared aggregates.

    uv run python -m training.report runs/<run>/traces.jsonl --name qwen3-4b-dev

Only the saved submission is scored; this never generates another model answer.
The saved episode reward must match the rescored reward. Failures stay in the
strict aggregate as zero, and the fixed split is checked before loading records.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from common import benchmark, config, metrics
from common.simulator import Simulator
from synthetic_hospital_dx.env import submission_from_text


def submission(trace: dict) -> dict | None:
    m = trace["metrics"]
    if not m.get("submitted") or m.get("malformed_submit"):
        return None
    for node in reversed(trace["nodes"]):
        message = node.get("message", {})
        if message.get("role") != "assistant":
            continue
        if m.get("submitted_as_text"):
            content = message.get("content", "")
            if isinstance(content, str):
                answer = submission_from_text(content)
                if answer is not None:
                    return answer
        else:
            for call in reversed(message.get("tool_calls", [])):
                if call["name"] == "submit_diagnosis":
                    return json.loads(call["arguments"])
    raise ValueError(f"Submitted trace {trace['id']} has no recoverable answer")


async def report(path: Path, split: str) -> dict:
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not episodes:
        raise ValueError("No episodes in trace file")
    if split == "public":
        # The paper's reported split: not part of our train/dev partition. Heldout is never loaded.
        instances = {i.gt_id: i for i in benchmark.load_instances(("public",))}
        allowed = set(instances)
    else:
        allowed = set(json.loads(config.SPLIT_FILE.read_text())[split])
        instances = {i.gt_id: i for i in benchmark.load_instances(("train",)) if i.gt_id in allowed}
    seen: set[int] = set()
    records, pairs, actions, token_counts, tool_counts = [], [], [], [], Counter()
    async with Simulator() as sim:
        for episode in episodes:
            if len(episode["traces"]) != 1:
                raise ValueError("This report expects one agent trace per patient")
            trace = episode["traces"][0]
            gt_id = int(episode["task"]["data"]["gt_id"])
            if gt_id not in allowed or gt_id in seen:
                raise ValueError(f"Unexpected or duplicate {split} patient gt_id={gt_id}")
            seen.add(gt_id)
            answer = submission(trace)
            reward_entry = trace["rewards"].get("official_score")
            if reward_entry is None and episode["ok"]:
                raise ValueError(f"Successful episode gt_id={gt_id} has no official reward")
            # Provider/harness failures may end before scoring. The evaluation
            # contract counts those episodes as zero, never drops them.
            reward = reward_entry["score"] if reward_entry is not None else 0.0
            scored = await sim.score(gt_id, answer) if answer is not None else None
            if scored is not None and not math.isclose(scored["reward"], reward, abs_tol=1e-9):
                raise ValueError(f"Saved/rescored reward differs for gt_id={gt_id}")
            usage = [call["usage"] for call in trace["calls"] if call.get("usage") is not None]
            input_tokens = sum(u["prompt_tokens"] for u in usage)
            output_tokens = sum(u["completion_tokens"] for u in usage)
            record = {
                "gt_id": gt_id, "n_pred": metrics.count_predictions(answer),
                "reward": reward, "metrics": scored["metrics"] if scored else None,
                "submission": answer, "episode_metrics": trace["metrics"],
                "ok": episode["ok"], "errors": [*episode["errors"], *trace["errors"]],
                "model_call_errors": [c["error"] for c in trace["calls"] if c.get("error")],
                # Provider totals count retransmitted history; Verifiers' totals do not.
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "unique_input_tokens": episode["num_input_tokens"],
                "unique_total_tokens": episode["num_total_tokens"],
                "budget": episode["task"]["data"]["budget"],
            }
            records.append(record)
            pairs.append((instances[gt_id], answer))
            if "actions" in trace["metrics"]:
                actions.append(trace["metrics"]["actions"])
            token_counts.append(record["total_tokens"])
            for node in trace["nodes"]:
                for call in node.get("message", {}).get("tool_calls", []):
                    tool_counts[call["name"]] += 1
    summary = metrics.summarize(records)
    summary["visit_split"] = metrics.split_recall(pairs)
    summary["split"] = split
    finalized = [r for r in records if r["episode_metrics"]]
    summary.update({
        "mean_actions": statistics.mean(actions) if actions else None,
        "actions_observed_n": len(actions),
        "episode_metrics_observed_n": len(finalized),
        "mean_total_tokens": statistics.mean(token_counts),
        "mean_input_tokens": statistics.mean(r["input_tokens"] for r in records),
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in records),
        "mean_unique_input_tokens": statistics.mean(r["unique_input_tokens"] for r in records),
        "mean_unique_total_tokens": statistics.mean(r["unique_total_tokens"] for r in records),
        "budget_exhausted_rate": statistics.mean(
            bool(r["episode_metrics"].get("forced_submit"))
            or r["episode_metrics"].get("actions", 0) >= r["budget"]
            for r in finalized
        ) if finalized else None,
        "malformed_submit_rate": statistics.mean(
            bool(r["episode_metrics"].get("malformed_submit")) for r in finalized
        ) if finalized else None,
        "rollout_error_rate": statistics.mean(not r["ok"] for r in records),
        "model_call_error_episode_rate": statistics.mean(bool(r["model_call_errors"]) for r in records),
        "model_call_error_types": dict(Counter(
            error["type"] for r in records for error in r["model_call_errors"]
        )),
        "mean_nudges": statistics.mean(r["episode_metrics"].get("nudges", 0) for r in finalized) if finalized else None,
        "tool_calls": dict(tool_counts),
    })
    models = sorted({call["model"] for e in episodes for call in e["traces"][0]["calls"]})
    return {"source": str(path), "split": split, "models": models,
            "complete_split": seen == allowed, "summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    args = parser.parse_args()
    result = asyncio.run(report(args.traces, args.split))
    result["summary"]["name"] = args.name
    target = config.write_result(f"training/{args.name}", result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
