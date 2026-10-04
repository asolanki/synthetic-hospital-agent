"""Markdown tables for the first-milestone results (results/*.json -> results/*.md)."""

from __future__ import annotations

import json

from shrl import config


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


def _with_visit_split(summary: dict, instances: dict) -> dict:
    """Attach the earlier-visit / final-visit recall split, computed from the saved trajectories."""
    from shrl import breakdown
    traj = config.RUNS_DIR / summary.get("name", "") / "trajectories.jsonl"
    if summary.get("name") and traj.exists():
        summary["visit_split"] = breakdown.for_run(traj, instances)
    return summary


def audit_table(result: dict) -> str:
    lines = [
        "| Strategy | Group | n | Mean reward | 95% CI | Official F1 | Official F1 (strict) | Precision "
        "| Weighted recall | Acuity acc | Mean # predictions |",
        "|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|",
    ]
    for r in result["rows"]:
        lo, hi = r["mean_reward_ci95"]
        lines.append(
            f"| {r['strategy']} | {r['group']} | {r['n']} | {_f(r['mean_reward'])} | {_f(lo)}–{_f(hi)} | "
            f"{_f(r['official_f1'])} | {_f(r['official_f1_strict'])} | {_f(r['precision_neutral'])} | "
            f"{_f(r['weighted_recall'])} | {_f(r['acuity_accuracy'])} | {r['mean_predictions']:.1f} |")
    return "\n".join(lines)


def build() -> str:
    parts = []
    baselines = [json.loads(p.read_text()) for p in sorted(config.RESULTS_DIR.glob("baseline_*.json"))
                 if not p.stem.startswith("baseline_smoke")]
    if baselines:
        from shrl import data
        instances = {i.gt_id: i for i in data.load_instances(("train", "public"))}
        baselines = [_with_visit_split(s, instances) for s in baselines]
        parts.append("## Agent baselines\n\n" + baseline_table(baselines))
    for path in sorted(config.RESULTS_DIR.glob("audit_*.json")):
        result = json.loads(path.read_text())
        parts.append(f"## Task/reward audit ({result['split']}, n={result['n_eval']})\n\n" + audit_table(result))
    text = "\n\n".join(parts) + "\n"
    (config.RESULTS_DIR / "tables.md").write_text(text)
    return text
