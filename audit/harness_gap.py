"""Paper agent harness with and without its coded problem-list label leak (patient_diagnosis).

Runs the upstream `eval/agents` structured harness unmodified, optionally masking the leak exactly
as /env does (outcome sections stripped, then `env_service.replace_problem_list`). Runs INSIDE the
simulator container, because it imports upstream `eval` and `epic_sim`. Every episode is scored with
the official `score_submission`; failures stay in the denominator at 0.

    uv run python -c "import json; from common import config; print(json.dumps(sorted(json.loads(config.SPLIT_FILE.read_text())['dev'])))" > /tmp/gt_ids.json
    docker cp /tmp/gt_ids.json shrl-app-1:/tmp/gt_ids.json
    docker cp audit/harness_gap.py shrl-app-1:/tmp/harness_gap.py
    set -a; source .env; set +a
    docker exec -e OPENROUTER_API_KEY shrl-app-1 python /tmp/harness_gap.py --model qwen3-30b \
        --gt-ids /tmp/gt_ids.json --out /tmp/qwen-paper.jsonl [--mask]
    docker exec shrl-app-1 python /tmp/harness_gap.py --model qwen3-30b \
        --gt-ids /tmp/gt_ids.json --out /tmp/qwen-paper.jsonl --summarize

The matching /env arm uses the same controls (`audit/harness_gap_env.toml`):

    uv run vf-eval @ audit/harness_gap_env.toml -n 120 -c 8 --no-rich \
        --output-dir runs/harness-check/env-qwen30b-t0 --run.dir eval
    uv run python -m training.report runs/harness-check/env-qwen30b-t0/eval/traces.jsonl --name ...

Results: `results/raw/audit/harness_gap_dev.json`.
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/app")
from eval.agents import runner, structured_agent  # noqa: E402
from eval.config import MODEL_REGISTRY, AdapterType, ModelConfig, get_pg_connection  # noqa: E402
from eval.score_one import _attach_context, load_instance, score_submission  # noqa: E402
from eval.scoring import compute_all_metrics  # noqa: E402
from epic_sim.app.services.env_service import _load_context, replace_problem_list  # noqa: E402

API = "http://localhost:8000"
# Our reference model, registered with the paper's OpenAI-compatible defaults (temperature 0,
# 16,384 max tokens). "mistral-3" is the paper's own registry entry (mistral-large-2512).
MODEL_REGISTRY["qwen3-30b"] = ModelConfig(
    name="qwen3-30b", adapter_type=AdapterType.OPENAI_COMPATIBLE,
    base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
    model_id="qwen/qwen3-30b-a3b-instruct-2507")
LOGS: dict[int, list] = {}


def match_key(code) -> str:
    norm = (code or "").upper().replace(".", "").strip()
    return norm[:5] if norm[:1] in ("S", "T") else norm[:3]


def coded_keys(obj) -> set[str]:
    """Match keys of every ICD-coded value anywhere in a tool result."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "icd" in k.lower() and isinstance(v, str) and v:
                found.add(match_key(v))
            else:
                found |= coded_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= coded_keys(v)
    return found


def predicted_keys(prediction) -> set[str]:
    if not isinstance(prediction, dict):
        return set()
    return {match_key(d.get("icd10")) for k in ("active_diagnoses", "chronic_conditions")
            for d in (prediction.get(k) or []) if isinstance(d, dict) and d.get("icd10")}


class Harness(structured_agent.StructuredAgentHarness):
    mask = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LOGS[self.gt_id] = self.tool_log
        self.chart_problems = _load_context(self.gt_id)["chart_problems"] if self.mask else None

    def call_tool(self, tool_name, arguments):
        result, entry = super().call_tool(tool_name, arguments)
        if self.mask:
            result = replace_problem_list(tool_name, result, self.chart_problems)
            entry["result"] = result
        return result, entry


runner.StructuredAgentHarness = Harness


def episode(model: str, gt_id: int, patient_id: int) -> dict:
    rec = {"gt_id": gt_id, "patient_id": patient_id, "model": model, "masked": Harness.mask}
    try:
        res = runner.run_agent_session(model, "structured", "patient_diagnosis", patient_id, gt_id,
                                       api_base=API, auth_token=runner.get_auth_token(API))
        rec.update(ok=True, prediction=res.prediction, submitted=res.submitted,
                   forced_submission=res.forced_submission, budget_exhausted=res.budget_exhausted,
                   turns=res.turns_used, input_tokens=res.total_input_tokens,
                   output_tokens=res.total_output_tokens)
    except Exception as exc:  # retained as a scored failure, never retried
        rec.update(ok=False, prediction=None, error=f"{type(exc).__name__}: {exc}")
    log = LOGS.pop(gt_id, [])
    rec["tools"] = [e["tool_name"] for e in log]
    rec["coded_keys_seen"] = sorted(set().union(*(coded_keys(e["result"]) for e in log)))
    rec["pred_keys"] = sorted(predicted_keys(rec["prediction"]))
    with get_pg_connection() as conn:
        scored = score_submission(conn, gt_id, rec["prediction"] or {})
        gt = load_instance(conn, gt_id)["ground_truth"]
    gold = {match_key(d.get("icd10_code") or d.get("icd10"))
            for k in ("active_diagnoses", "chronic_conditions") for d in gt.get(k, [])}
    rec["reward"], rec["metrics"] = scored["reward"], scored["metrics"]
    rec["gold_keys"] = len(gold)
    rec["gold_keys_seen_in_tools"] = len(gold & set(rec["coded_keys_seen"]))
    rec["gold_keys_predicted"] = len(gold & set(rec["pred_keys"]))
    return rec


def summarize(path: str, gt_ids: list[int]) -> dict:
    rows = {r["gt_id"]: r for r in map(json.loads, open(path))}
    missing = sorted(set(gt_ids) - set(rows))
    preds, raw_gts, context_gts = [], [], []
    with get_pg_connection() as conn:
        for gt_id in gt_ids:
            if gt_id in rows:
                inst = load_instance(conn, gt_id)
                preds.append(rows[gt_id]["prediction"] or {})
                raw_gts.append(inst["ground_truth"])
                context_gts.append(_attach_context(conn, inst))
    n = len(preds)
    used = lambda tool: sum(tool in r["tools"] for r in rows.values()) / n
    gold_total = sum(r["gold_keys"] for r in rows.values())
    return {
        "n": n, "missing": missing,
        "mean_episode_reward": sum(r["reward"] for r in rows.values()) / n,
        "paper_runner_batch_f1": compute_all_metrics("patient_diagnosis", preds, raw_gts)[
            "weighted_problem_list_f1_neutral"],
        "batch_f1_with_neutral_context": compute_all_metrics("patient_diagnosis", preds, context_gts)[
            "weighted_problem_list_f1_neutral"],
        "submitted_rate": sum(bool(r.get("submitted")) for r in rows.values()) / n,
        "error_rate": sum(not r["ok"] for r in rows.values()) / n,
        "mean_tool_calls": sum(len(r["tools"]) for r in rows.values()) / n,
        "used_view_problem_list": used("view_problem_list"),
        "used_open_chart": used("open_chart"),
        "gold_keys_seen_in_tools": sum(r["gold_keys_seen_in_tools"] for r in rows.values()) / gold_total,
        "gold_key_recall": sum(r["gold_keys_predicted"] for r in rows.values()) / gold_total,
        "mean_input_tokens": sum(r.get("input_tokens") or 0 for r in rows.values()) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen3-30b", "mistral-3"), required=True)
    ap.add_argument("--gt-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    gt_ids = json.load(open(args.gt_ids))[: args.limit]
    if args.summarize:
        print(json.dumps(summarize(args.out, gt_ids), indent=1))
        return
    Harness.mask = args.mask
    try:
        done = {json.loads(line)["gt_id"] for line in open(args.out)}
    except FileNotFoundError:
        done = set()
    with get_pg_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT gt_id, patient_id FROM benchmark_ground_truth WHERE gt_id = ANY(%s)", (gt_ids,))
        patients = dict(cur.fetchall())
    todo = [g for g in gt_ids if g not in done]
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a") as out:
        futures = [pool.submit(episode, args.model, g, patients[g]) for g in todo]
        for i, future in enumerate(as_completed(futures), 1):
            rec = future.result()
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            print(f"{i}/{len(todo)} gt={rec['gt_id']} ok={rec['ok']} reward={rec['reward']:.3f} "
                  f"tools={len(rec['tools'])} gold_seen={rec['gold_keys_seen_in_tools']}/{rec['gold_keys']}",
                  flush=True)


if __name__ == "__main__":
    main()
