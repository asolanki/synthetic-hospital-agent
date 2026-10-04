"""Rebuild results/tables.md (milestone-1 tables) from results/raw/.

    uv run python -m baselines.report

Baseline rows get the earlier-visit / final-visit recall split, computed from the
saved trajectories in runs/<name>/ when they exist locally.
"""

from __future__ import annotations

import json
from pathlib import Path

from audit.strategies import audit_table
from common import benchmark, config, metrics


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _f(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def baseline_table(summaries: list[dict]) -> str:
    lines = [
        "| Model | Split | n | Agent F1 (mean reward) | 95% CI | Official F1 (strict) | Precision | Weighted recall "
        "| Recall, earlier visits | Recall, final visit "
        "| Acuity acc | Valid submit % | Actions | Tokens | Budget exhausted % |",
        "|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for s in summaries:
        lo, hi = s["mean_reward_ci95"]
        split = s.get("visit_split") or {}
        lines.append(
            f"| {s.get('name') or s['config']['model']} | {s.get('split', '')} | {s['n']} | {_f(s['mean_reward'])} | "
            f"{_f(lo)}–{_f(hi)} | {_f(s['official_f1_strict'])} | {_f(s['precision_neutral'])} | "
            f"{_f(s['weighted_recall'])} | {_f(split.get('recall_earlier_visits'))} | "
            f"{_f(split.get('recall_final_visit'))} | {_f(s['acuity_accuracy'])} | {_pct(s['valid_submit_rate'])} | "
            f"{s['mean_actions']:.1f} | {s['mean_total_tokens']:,.0f} | {_pct(s['budget_exhausted_rate'])} |")
    return "\n".join(lines)


def visit_split_for_run(traj_path: Path, instances: dict[int, benchmark.Instance]) -> dict:
    """Earlier-visit / final-visit recall of a baselines/agent.py run (latest attempt per patient)."""
    latest: dict[int, dict] = {}
    for line in traj_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["gt_id"]] = rec
    return metrics.split_recall([(instances[r["gt_id"]], r["submission"]) for r in latest.values()
                                 if r["stop_reason"] != "llm_error" and r["gt_id"] in instances])


def build() -> str:
    parts = []
    baselines = [json.loads(p.read_text())
                 for p in sorted((config.RAW_RESULTS_DIR / "baselines").glob("*.json"))]
    if baselines:
        instances = {i.gt_id: i for i in benchmark.load_instances(("train", "public"))}
        for s in baselines:
            traj = config.RUNS_DIR / s.get("name", "") / "trajectories.jsonl"
            if s.get("name") and traj.exists():
                s["visit_split"] = visit_split_for_run(traj, instances)
        parts.append("## Agent baselines\n\n" + baseline_table(baselines))
    for path in sorted((config.RAW_RESULTS_DIR / "audit").glob("strategies_*.json")):
        result = json.loads(path.read_text())
        parts.append(f"## Task/reward audit ({result['split']}, n={result['n_eval']})\n\n" + audit_table(result))
    text = "\n\n".join(parts) + "\n"
    (config.RESULTS_DIR / "tables.md").write_text(text)
    return text


if __name__ == "__main__":
    print(build())
