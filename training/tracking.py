"""W&B tables, transcript media and artifacts for the canonical evaluation reports.

    uv run --group tracking python -m training.tracking REPORT --traces TRACES --config CONFIG

Raw files remain the source of truth. Incremental tables show completed episodes;
these are not token-by-token live traces or replacement scoring calculations.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import tomllib

import wandb

from common import config


def tracking_settings() -> dict:
    settings = tomllib.loads((config.ROOT / "training/configs/sft_oracle_copy.toml").read_text())
    return {"project": settings["wandb_project"], "entity": settings["wandb_entity"]}


def transcript(episode: dict) -> wandb.Html:
    sections = []
    if episode.get("errors"):
        body = html.escape(json.dumps(episode["errors"], ensure_ascii=False, indent=2))
        sections.append(f"<details open><summary>Episode errors</summary><pre>{body}</pre></details>")
    for trace in episode["traces"]:
        if trace.get("errors"):
            body = html.escape(json.dumps(trace["errors"], ensure_ascii=False, indent=2))
            sections.append(f"<details open><summary>Trace errors</summary><pre>{body}</pre></details>")
        for node in trace["nodes"]:
            message = node.get("message")
            if message is None:
                continue
            role = html.escape(message.get("role", "message"))
            body = html.escape(json.dumps(message, ensure_ascii=False, indent=2))
            sections.append(f"<details open><summary>{role}</summary><pre>{body}</pre></details>")
    return wandb.Html(
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        'body{font-family:system-ui;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'details{margin-bottom:16px}summary{font-weight:bold}</style></head><body>'
        + "\n".join(sections) + "</body></html>"
    )


class RolloutLog:
    """Log decoded episodes or consume complete appended JSONL records."""

    def __init__(self, run, path: Path | None = None):
        self.run = run
        self.path = path
        self.offset = 0
        self.count = 0
        self.reward_sum = 0.0
        self.pending = 0
        self.table = wandb.Table(columns=[
            "gt_id", "reward", "episode_ok", "model_call_errors", "submitted",
            "actions", "nudges", "output_tokens", "transcript",
            "episode_id", "dispatch_step", "policy_start", "policy_end",
        ], log_mode="INCREMENTAL")
        run.define_metric("progress/*", step_metric="progress/completed")

    def add_episode(self, episode: dict) -> None:
        traces = episode["traces"]
        if len(traces) > 1 or (episode["ok"] and len(traces) != 1):
            raise ValueError("Hospital tracking expects one trace, except for early failures")
        trace = traces[0] if traces else {}
        reward_entry = trace.get("rewards", {}).get("official_score")
        if reward_entry is None and episode["ok"]:
            raise ValueError("Successful completed episode has no official reward")
        reward = reward_entry["score"] if reward_entry is not None else 0.0
        metrics = trace.get("metrics", {})
        work = (episode.get("run") or {}).get("work", {})
        policy = work.get("policy") or {}
        self.table.add_data(
            int(episode["task"]["data"]["gt_id"]), reward, episode["ok"],
            sum(bool(call.get("error")) for call in trace.get("calls", [])),
            metrics.get("submitted"), metrics.get("actions"), metrics.get("nudges"),
            episode["num_output_tokens"], transcript(episode), episode["id"],
            work.get("step"), policy.get("start"), policy.get("end"),
        )
        self.count += 1
        self.reward_sum += reward
        self.pending += 1

    def flush(self) -> None:
        if self.pending:
            self.run.log({"rollouts": self.table, "progress/completed": self.count,
                          "progress/mean_reward": self.reward_sum / self.count})
            self.pending = 0

    def update(self) -> None:
        if self.path is None:
            raise ValueError("JSONL updates require a file path")
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                episode = json.loads(line)
                self.add_episode(episode)
                self.offset = stream.tell()
        self.flush()


def numeric_metrics(values: dict, prefix: str) -> dict:
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            result.update(numeric_metrics(value, name))
        elif isinstance(value, (int, float)):
            result[name] = value
        elif key.endswith("_ci95") and isinstance(value, list) and len(value) == 2:
            result[f"{name}/lower"], result[f"{name}/upper"] = value
    return result


def publish_report(run, report_path: Path, trace_path: Path, config_path: Path) -> None:
    result = json.loads(report_path.read_text())
    if not result["complete_split"]:
        raise ValueError("Only complete fixed-split reports are published as evaluation results")
    run.log(numeric_metrics(result["summary"], "eval"))
    run.summary["complete_split"] = True
    run.summary["models"] = result["models"]
    if "performance" in result:
        run.log(numeric_metrics(result["performance"], "performance"))
    artifact = wandb.Artifact(run.name, type="evaluation")
    for path, name in ((report_path, "report.json"), (trace_path, "traces.jsonl"),
                       (config_path, "config.json")):
        artifact.add_file(str(path), name=name)
    run.log_artifact(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    result = json.loads(args.report.read_text())
    if not result["complete_split"]:
        parser.error("Cannot publish an incomplete evaluation as a fixed-split result")
    with wandb.init(
        **tracking_settings(), name=result["summary"]["name"], group="qwen3-4b",
        job_type="evaluation", config=json.loads(args.config.read_text()), mode=args.mode,
        dir=str(config.ROOT / "runs"),
        tags=["backfilled"], settings=wandb.Settings(disable_git=True, save_code=False),
    ) as run:
        RolloutLog(run, args.traces).update()
        publish_report(run, args.report, args.traces, args.config)
        print(json.dumps({"wandb_url": run.url}), flush=True)


if __name__ == "__main__":
    main()
