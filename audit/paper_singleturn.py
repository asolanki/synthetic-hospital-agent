"""Paper's single-turn patient_diagnosis run for Gemma 3 27B, with the paper's own code.

Reproduces Table 3's single-turn Gemma 3 27B number (severity-weighted F1 0.287, public split,
200 patients). Runs INSIDE the simulator container and calls the upstream pipeline unmodified:
`eval.runner.run_evaluation` (loader with assessment/plan hidden, locked CoT prompt, adapter,
parser, Postgres persistence) and `eval.runner.rescore_run` (= `python -m eval.cli score`).
The only change is the `gemma-3` registry entry: DeepInfra's OpenAI-compatible API directly
instead of OpenRouter pinned to DeepInfra, so the OpenRouter-only `provider` field is dropped.
Model id, temperature (0), max_tokens (16,384) and timeout keep the paper's defaults.

Calls that exhaust their retries are not stored by upstream; the driver re-enters the same run
with upstream's resume path (`--resume`) up to `--passes` times, then reports what is left as
failures. Retries use upstream's env-gated knobs (EVAL_MAX_RETRIES / EVAL_RETRY_MAX_BACKOFF).

    docker cp audit/paper_singleturn.py shrl-app-1:/tmp/paper_singleturn.py
    set -a; source .env; set +a
    docker exec -e DEEPINFRA_API_KEY shrl-app-1 python /tmp/paper_singleturn.py --pilot 3 --out /tmp/smoke.json
    docker exec -e DEEPINFRA_API_KEY shrl-app-1 python /tmp/paper_singleturn.py --out /tmp/gemma.json
    docker cp shrl-app-1:/tmp/gemma.json results/raw/audit/paper_singleturn_gemma3_public.json
    docker exec shrl-app-1 python /tmp/paper_singleturn.py --report <run_id> --out /tmp/gemma.json

Results: `results/raw/audit/paper_singleturn_gemma3_public.json`.
"""
import argparse
import dataclasses
import json
import logging
import os
import sys

sys.path.insert(0, "/app")
os.environ.setdefault("EVAL_MAX_RETRIES", "8")
os.environ.setdefault("EVAL_RETRY_MAX_BACKOFF", "60")

from eval.config import MODEL_REGISTRY, COST_RATES, ModelConfig, estimate_cost_usd, get_pg_connection  # noqa: E402
from eval.runner import rescore_run, run_evaluation  # noqa: E402
from eval.score_one import _attach_context, load_instance, score_submission  # noqa: E402
from eval.scoring import compute_all_metrics  # noqa: E402
from eval.tasks.patient_diagnosis import load_inputs  # noqa: E402
from etl.stages.s05_ontology import download_icd10  # noqa: E402

TASK, MODEL, SPLIT = "patient_diagnosis", "gemma-3", "public"
STRATEGY = "cot"  # eval/strategy_experiment.py LOCKED_STRATEGIES["patient_diagnosis"]; README
DEEPINFRA_RATES = (0.08, 0.16)  # USD per 1M input/output tokens, google/gemma-3-27b-it
# Stands in for a failed or empty prediction in the strict batch: matches no reference and no
# chart-neutral category, so the item contributes recall 0 and precision 0 instead of being skipped.
NEVER_MATCHES = {"active_diagnoses": [{"icd10": "ZZZ.9", "acuity": "acute"}], "chronic_conditions": []}

PAPER_ENTRY = dataclasses.asdict(MODEL_REGISTRY[MODEL])
MODEL_REGISTRY[MODEL] = dataclasses.replace(
    MODEL_REGISTRY[MODEL], base_url="https://api.deepinfra.com/v1/openai",
    api_key_env="DEEPINFRA_API_KEY", extra={})
assert isinstance(MODEL_REGISTRY[MODEL], ModelConfig)

ERRORS: dict[int, str] = {}


class FailureLog(logging.Handler):
    """Collects upstream's `Failed gt_id=%d: %s` records (run_evaluation logs, never stores, them)."""

    def emit(self, record):
        if record.msg.startswith("Failed gt_id=") and len(record.args) == 2:
            gt_id, err = record.args
            cause = err.__cause__
            ERRORS[gt_id] = f"{err}" + (f" <- {type(cause).__name__}: {cause}" if cause else "")


def has_dx(pred) -> bool:
    return isinstance(pred, dict) and bool(
        pred.get("active_diagnoses") or pred.get("chronic_conditions") or pred.get("diagnoses"))


def report(run_id: int, cli_metrics: dict, passes: list[dict], pilot: int | None) -> dict:
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT run_name, config, started_at, completed_at FROM evaluation_runs"
                        " WHERE run_id = %s", (run_id,))
            run_name, run_config, created, completed = cur.fetchone()
            cur.execute("SELECT gt_id, prediction, raw_output, input_tokens, output_tokens, latency_ms"
                        " FROM evaluation_predictions WHERE run_id = %s", (run_id,))
            stored = {r[0]: r[1:] for r in cur.fetchall()}
        gt_ids = [inp.gt_id for inp in load_inputs(conn, split=SPLIT, pilot=pilot)]

        rows, preds, gts, strict_preds, strict_gts = [], [], [], [], []
        for gt_id in gt_ids:
            inst = load_instance(conn, gt_id)
            gt = _attach_context(conn, inst)
            if gt_id in stored:
                pred, raw, tin, tout, lat = stored[gt_id]
                pred = pred if isinstance(pred, dict) else json.loads(pred)
                scored = score_submission(conn, gt_id, pred, task=TASK)
                error = None if has_dx(pred) else f"empty prediction (parse tier {pred.get('_parse_tier')})"
                preds.append(pred)
                gts.append(gt)
            else:
                pred = raw = tin = tout = lat = scored = None
                error = ERRORS.get(gt_id, "no stored prediction (call failed; error not captured in this process)")
            strict_preds.append(pred if has_dx(pred) else NEVER_MATCHES)
            strict_gts.append(gt)
            m = (scored or {}).get("metrics", {})
            rows.append({
                "gt_id": gt_id, "patient_id": inst["patient_id"],
                "raw_output": raw, "raw_output_truncated_at_10000": bool(raw) and len(raw) >= 10000,
                "parsed_prediction": pred,
                "parse_tier": (pred or {}).get("_parse_tier"),
                "parseable": has_dx(pred),
                "reward": scored["reward"] if scored else 0.0,
                "reward_metric": "weighted_problem_list_f1_neutral",
                "weighted_problem_list_recall": m.get("weighted_problem_list_recall"),
                "problem_list_precision_neutral": m.get("problem_list_precision_neutral"),
                "input_tokens": tin, "output_tokens": tout, "latency_ms": lat,
                "error": error,
            })
        official_neutral = compute_all_metrics(TASK, preds, gts)
        strict = compute_all_metrics(TASK, strict_preds, strict_gts)
    finally:
        conn.close()

    tin = sum(r["input_tokens"] or 0 for r in rows)
    tout = sum(r["output_tokens"] or 0 for r in rows)
    tiers: dict[str, int] = {}
    for r in rows:
        tiers[r["parse_tier"] or "no_prediction"] = tiers.get(r["parse_tier"] or "no_prediction", 0) + 1
    rewards = [r["reward"] for r in rows]
    return {
        "question": "Does the paper's own single-turn pipeline reproduce Table 3's Gemma 3 27B "
                    "patient_diagnosis value (severity-weighted F1 0.287, public split)?",
        "paper_value": 0.287,
        "run_id": run_id, "run_name": run_name, "started_at": str(created), "completed_at": str(completed),
        "n_attempted": len(gt_ids),
        "n_stored": sum(r["gt_id"] in stored for r in rows),
        "n_parsed": sum(r["parseable"] for r in rows),
        "n_failed_or_empty": sum(not r["parseable"] for r in rows),
        "parse_tiers": tiers,
        "official_f1": {
            "note": "Paper's batch scorer (compute_all_metrics); skips empty predictions (n_tier_c) "
                    "and calls that never stored a prediction.",
            "cli_score_as_released": {
                "how": "eval.runner.rescore_run == `python -m eval.cli score`; the released batch path "
                       "does not attach chart-neutral sets, so *_neutral equals the non-neutral value",
                "weighted_problem_list_f1": cli_metrics.get("weighted_problem_list_f1"),
                "weighted_problem_list_f1_neutral": cli_metrics.get("weighted_problem_list_f1_neutral"),
                "metrics": cli_metrics,
            },
            "chart_neutral_primary": {
                "how": "same scorer with each GT given its chart-neutral set via score_one._attach_context "
                       "(the README's primary rule, weighted_problem_list_f1_neutral)",
                "weighted_problem_list_f1_neutral": official_neutral["weighted_problem_list_f1_neutral"],
                "metrics": official_neutral,
            },
        },
        "strict_f1": {
            "how": "same scorer with chart-neutral sets, all attempted patients; failed or empty "
                   "predictions replaced by a never-matching code so they count recall 0, precision 0",
            "weighted_problem_list_f1_neutral": strict["weighted_problem_list_f1_neutral"],
            "weighted_problem_list_f1": strict["weighted_problem_list_f1"],
            "metrics": strict,
        },
        "mean_reward": {
            "how": "mean per-patient score_submission reward (weighted_problem_list_f1_neutral), "
                   "failures 0, over all attempted patients",
            "value": sum(rewards) / len(rewards) if rewards else 0.0,
        },
        "tokens": {"input": tin, "output": tout},
        "cost_usd": {
            "deepinfra_rates": (tin * DEEPINFRA_RATES[0] + tout * DEEPINFRA_RATES[1]) / 1e6,
            "deepinfra_rates_per_1m": DEEPINFRA_RATES,
            "paper_cost_rates": estimate_cost_usd(MODEL, tin, tout),
            "paper_cost_rates_per_1m": COST_RATES[MODEL],
            "note": "stored predictions only; retried/failed calls are not counted",
        },
        "settings": {
            "task": TASK, "split": SPLIT, "prompt_strategy": STRATEGY,
            "strategy_source": "eval/strategy_experiment.py LOCKED_STRATEGIES; README 'Locked prompting strategies'",
            "model_config_used": {k: v for k, v in dataclasses.asdict(MODEL_REGISTRY[MODEL]).items()},
            "paper_registry_entry": PAPER_ENTRY,
            "run_config_stored": run_config,
            "retry_env": {k: os.environ.get(k) for k in
                          ("EVAL_MAX_RETRIES", "EVAL_RETRY_BASE_DELAY", "EVAL_RETRY_MAX_BACKOFF")},
            "passes": passes,
            "deviations": [
                "Endpoint: DeepInfra direct instead of OpenRouter pinned to DeepInfra; OpenRouter's "
                "`provider` routing field dropped (not an OpenAI parameter).",
                "Retries: EVAL_MAX_RETRIES and EVAL_RETRY_MAX_BACKOFF raised via upstream's env knobs.",
                "Failed calls re-entered with upstream's resume path within the same run_id.",
                "Setup, not a code change: the container image lacks data/ontology/icd10cm_2025.csv, "
                "which the parser's ICD-10 validator requires; created with upstream's "
                "etl.stages.s05_ontology.download_icd10 (CMS 2025 release).",
            ],
        },
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=None, help="first N public patients (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", type=int, default=3, help="resume passes over failed calls")
    ap.add_argument("--resume", type=int, default=None, help="continue an existing run_id")
    ap.add_argument("--report", type=int, default=None, help="only write the report for run_id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("eval.runner").addHandler(FailureLog())
    # The parser snaps invalid codes to ICD-10-CM 2025 (eval/icd10_validator.py), which reads
    # /app/data/ontology/icd10cm_2025.csv; the image ships without it. Upstream's own helper
    # fetches and parses the CMS release (no-op if the file exists).
    download_icd10()

    passes = []
    run_id = args.report or args.resume
    if args.report is None:
        conn = get_pg_connection()
        try:
            n_total = len(load_inputs(conn, split=SPLIT, pilot=args.pilot))
        finally:
            conn.close()
        for i in range(args.passes):
            before = dict(ERRORS)
            ERRORS.clear()
            run_id = run_evaluation(TASK, MODEL, split=SPLIT, prompt_strategy=STRATEGY,
                                    pilot=args.pilot, workers=args.workers, resume_run_id=run_id)
            passes.append({"pass": i + 1, "run_id": run_id, "failed": sorted(ERRORS)})
            for k, v in before.items():
                ERRORS.setdefault(k, v)
            conn = get_pg_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM evaluation_predictions WHERE run_id = %s", (run_id,))
                    done = cur.fetchone()[0]
            finally:
                conn.close()
            if done >= n_total:
                break
    cli_metrics = rescore_run(run_id)
    out = report(run_id, cli_metrics, passes, args.pilot)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: out[k] for k in ("run_id", "n_attempted", "n_parsed", "n_failed_or_empty",
                                          "parse_tiers", "tokens", "cost_usd")}, default=str))
    print("cli score weighted F1:", out["official_f1"]["cli_score_as_released"]["weighted_problem_list_f1"])
    print("chart-neutral official F1:", out["official_f1"]["chart_neutral_primary"]["weighted_problem_list_f1_neutral"])
    print("strict F1:", out["strict_f1"]["weighted_problem_list_f1_neutral"])
    print("mean reward:", out["mean_reward"]["value"])


if __name__ == "__main__":
    main()
