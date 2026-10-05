"""Summarize a completed native RL run without confusing arrivals with updates.

Run with the pinned Prime-RL Python environment from the repository root:
    python -m training.rl_report runs/rl/NAME --expected-steps 85 --name NAME-training

Arrival records alone do not contain every exclusion reason. Preserve native
windowed cancellation/error/admission counters separately; they are not a
mutually exclusive partition of all finalized episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from prime_rl.utils.pathing import get_file_monitor_dir

from common import config
from training.track_rl import read_appended


def summarize(run_dir: Path, expected_steps: int, batch_size: int = 32) -> dict:
    train_ids = set(json.loads(config.SPLIT_FILE.read_text())["train"])
    shipped = {}
    for update in read_appended(get_annotations_dir(run_dir) / "orch", {}):
        info = update["info"]
        if info.get("effective"):
            trace_id = update["trace_id"]
            if trace_id in shipped:
                raise ValueError(f"Trace shipped more than once: {trace_id}")
            shipped[trace_id] = info

    records, identity = [], None
    observed_ids, observed_shipped = set(), set()
    errors = Counter()
    for episode in read_appended(get_trace_stream(run_dir), {}):
        gt_id = int(episode["task"]["data"]["gt_id"])
        if gt_id not in train_ids or episode["id"] in observed_ids:
            raise ValueError("Unexpected patient or duplicate episode in training stream")
        observed_ids.add(episode["id"])
        if episode["run"]["type"] != "train":
            raise ValueError("Expected native training-run provenance")
        current = {key: episode["run"][key] for key in ("id", "name")}
        if identity is not None and identity != current:
            raise ValueError("Multiple training runs in one stream")
        identity = current
        traces = episode["traces"]
        if len(traces) > 1 or (episode["ok"] and not traces):
            raise ValueError("Expected one hospital agent trace per successful episode")
        trace = traces[0] if traces else {}
        reward = trace.get("rewards", {}).get("official_score")
        if reward is None and episode["ok"]:
            raise ValueError("Successful episode has no official reward")
        info = shipped.get(trace.get("id"))
        policy = episode["run"]["work"]["policy"]
        if info is not None:
            step = info["ship"]["step"]
            if not episode["ok"] or policy != {"start": step - 1, "end": step - 1}:
                raise ValueError("Errored, stale or mixed-policy trace reached training")
            observed_shipped.add(trace["id"])
        error_types = {error["type"] for error in [*episode["errors"], *trace.get("errors", [])]}
        errors.update(error_types)
        records.append({"gt_id": gt_id, "reward": reward["score"] if reward else 0.0,
                        "ok": episode["ok"], "group": episode["group"]["id"],
                        "step": info["ship"]["step"] if info else None,
                        "output_tokens": episode["num_output_tokens"],
                        "actions": trace.get("metrics", {}).get("actions")})
    if observed_shipped != set(shipped):
        raise ValueError("Shipped traces are missing from the arrival stream")
    effective = [row for row in records if row["step"] is not None]
    counts = Counter(row["step"] for row in effective)
    if counts != Counter({step: batch_size for step in range(1, expected_steps + 1)}):
        raise ValueError(f"Incomplete or unexpected training batches: {counts}")

    curves = []
    keys = {"step", "time", "producer", "loss/mean", "optim/grad_norm",
            "mismatch_kl/all/mean", "time/step", "time/wait_for_batch",
            "time/forward_backward", "time/save_ckpt", "time/broadcast_weights",
            "progress/rollouts", "progress/tasks", "off_policy/dropped",
            "train/agg/all/has_error/mean", "train/agg/all/cancelled/mean",
            "train/agg/all/agent/is_admitted/mean",
            "train/agg/all/agent/reward/mean", "train/agg/effective/agent/reward/mean"}
    metric_path = get_file_monitor_dir(run_dir) / "metrics.jsonl"
    for line in metric_path.read_text().splitlines():
        row = json.loads(line)
        if "loss/mean" in row or "progress/rollouts" in row:
            curves.append({key: value for key, value in row.items() if key in keys})
    trainer_steps = [row["step"] for row in curves if "loss/mean" in row]
    if sorted(trainer_steps) != list(range(1, expected_steps + 1)):
        raise ValueError("Terminal trainer metrics do not cover every expected update")

    def cohort(rows: list[dict]) -> dict:
        patients = Counter(row["gt_id"] for row in rows)
        actions = [row["actions"] for row in rows if row["actions"] is not None]
        return {"episodes": len(rows), "unique_patients": len(patients),
                "unique_groups": len({row["group"] for row in rows}),
                "mean_reward": statistics.mean(row["reward"] for row in rows),
                "errored_episodes": sum(not row["ok"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "mean_actions": statistics.mean(actions) if actions else None,
                "episodes_per_patient": dict(sorted(patients.items())),
                "unseen_train_patients": sorted(train_ids - patients.keys())}

    grouped = defaultdict(list)
    for row in effective:
        grouped[row["step"]].append(row)
    return {
        "run": identity, "source": str(run_dir), "expected_steps": expected_steps,
        "qualification": "Training distributions, not dev performance. All-finalized and shipped cohorts differ; native windowed exclusion counters can overlap and do not partition the arrival stream.",
        "all_finalized": cohort(records), "shipped_to_trainer": cohort(effective),
        "finalized_not_shipped": len(records) - len(effective),
        "episode_error_types": dict(errors), "shipped_policy_staleness": 0,
        "per_update": [{"step": step, "episodes": len(rows),
                        "unique_patients": len({row["gt_id"] for row in rows}),
                        "mean_reward": statistics.mean(row["reward"] for row in rows)}
                       for step, rows in sorted(grouped.items())],
        "native_curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir, args.expected_steps, args.batch_size)
    path = config.write_result(f"training/{args.name}", result)
    print(json.dumps({key: result[key] for key in ("run", "expected_steps", "finalized_not_shipped", "episode_error_types")}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
