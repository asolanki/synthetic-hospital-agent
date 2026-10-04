"""Audit of the benchmark's other official tasks: is chart interaction the bottleneck?

Same questions the patient_diagnosis audit asked, for `evidence_retrieval` and
`imaging_indication`, on the dev patients (the 120 patients of the fixed split):

  - does the oracle score 1.0;
  - what do hand-written strategies that barely look at the chart score;
  - (imaging) how much does a model gain from seeing nothing, the current note,
    or the whole prior chart, and does a tool-using agent reach the same level.

    uv run python -m audit.other_tasks retrieval
    uv run python -m audit.other_tasks summarization
    uv run python -m audit.other_tasks imaging-strategies
    uv run python -m audit.other_tasks imaging-llm --model qwen/qwen3-30b-a3b-instruct-2507
    uv run python -m audit.other_tasks imaging-agent --model qwen/qwen3-30b-a3b-instruct-2507

Results go to results/raw/other_tasks/. `imaging-agent` reuses the early tool-use
harness in baselines/agent.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re

import openai

from baselines import agent
from common import benchmark, config, splits
from common.simulator import Simulator

SECTION_PRIOR = ["hpi", "pathology", "imaging", "other_studies", "physical_exam", "chief_complaint", "ros",
                 "labs", "pmh", "psh", "family_history", "social_history", "vitals", "medications", "allergies"]


def dev_instances() -> dict[int, benchmark.Instance]:
    """Dev patients keyed by patient_id (charts loaded, outcome sections removed)."""
    dev = set(splits.load()["dev"])
    return {i.patient_id: i for i in benchmark.load_instances(("train",)) if i.gt_id in dev}


def task_rows(task: str, patients: dict[int, benchmark.Instance]) -> list[dict]:
    conn = benchmark.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT gt_id, patient_id, encounter_id, ground_truth FROM benchmark_ground_truth "
        "WHERE task = ? AND is_diagnostic AND split = 'train' ORDER BY gt_id", (task,))
        if r["patient_id"] in patients]
    for r in rows:
        r["ground_truth"] = json.loads(r["ground_truth"])
    orders = {r["gt_id"]: dict(r) for r in conn.execute("SELECT * FROM imaging_orders")}
    for r in rows:
        r["order"] = orders.get(r["gt_id"])
    conn.close()
    return rows


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def write(name: str, payload: dict) -> None:
    config.write_result(f"other_tasks/{name}", payload)


# ---------------------------------------------------------------------------
# evidence_retrieval
# ---------------------------------------------------------------------------

async def retrieval() -> None:
    patients = dev_instances()
    rows = task_rows("evidence_retrieval", patients)
    rng = random.Random(0)

    def sections(pid: int) -> list[dict]:
        return [s for e in patients[pid].encounters for s in e["sections"]]

    def ranked(secs: list[dict]) -> dict:
        return {"rankings": [{"passage_id": f"ees_{s['section_id']}", "grade": 3} for s in secs]}

    strategies = {
        "empty": lambda r: {"rankings": []},
        "5 random sections": lambda r: ranked(rng.sample(sections(r["patient_id"]), 5)),
        "first 5 sections of the chart": lambda r: ranked(sections(r["patient_id"])[:5]),
        "all sections, chart order": lambda r: ranked(sections(r["patient_id"])),
        "all sections, sorted by section type (HPI first)": lambda r: ranked(sorted(
            sections(r["patient_id"]), key=lambda s: SECTION_PRIOR.index(s["section_type"])
            if s["section_type"] in SECTION_PRIOR else len(SECTION_PRIOR))),
    }
    out = {"n": len(rows), "rows": []}
    async with Simulator() as sim:
        for name, fn in strategies.items():
            preds = [fn(r) for r in rows]
            res = await sim.score_batch([(r["gt_id"], p) for r, p in zip(rows, preds)], task="evidence_retrieval")
            out["rows"].append({"strategy": name, "precision_5 (reward)": mean(x["reward"] for x in res),
                                "ndcg_10": mean(x["metrics"]["ndcg_10"] for x in res)})
    write("retrieval_dev", out)
    print(f"evidence_retrieval, dev, n={len(rows)}")
    for r in out["rows"]:
        print(f"  {r['strategy']:52s} P@5={r['precision_5 (reward)']:.3f}  nDCG@10={r['ndcg_10']:.3f}")


# ---------------------------------------------------------------------------
# context_summarization (whole-patient variant)
# ---------------------------------------------------------------------------

async def summarization() -> None:
    patients = dev_instances()
    rows = [r for r in task_rows("context_summarization", patients)
            if r["ground_truth"].get("variant", "unconditioned") == "unconditioned" and r["encounter_id"] is None]

    def sections(pid: int, types: tuple[str, ...] | None = None) -> str:
        return "\n".join(s["section_text"] or "" for e in patients[pid].encounters for s in e["sections"]
                         if types is None or s["section_type"] in types)

    strategies = {
        "empty": lambda r: "",
        "chief complaints only": lambda r: ". ".join(e["chief_complaint"] or "" for e in patients[r["patient_id"]].encounters),
        "every HPI, concatenated": lambda r: sections(r["patient_id"], ("hpi",)),
        "HPI + exam + labs + imaging, concatenated": lambda r: sections(
            r["patient_id"], ("hpi", "physical_exam", "labs", "imaging", "other_studies", "pathology", "vitals")),
        "the entire chart, pasted": lambda r: sections(r["patient_id"]),
    }
    out = {"n": len(rows), "rows": []}
    async with Simulator() as sim:
        for name, fn in strategies.items():
            preds = [{"summary": fn(r)} for r in rows]
            res = await sim.score_batch([(r["gt_id"], p) for r, p in zip(rows, preds)], task="context_summarization")
            out["rows"].append({"strategy": name, "finding_recall (reward)": mean(x["reward"] for x in res),
                                "mean_words": mean(len(p["summary"].split()) for p in preds)})
    write("summarization_dev", out)
    print(f"context_summarization (whole-patient), dev, n={len(rows)}")
    for r in out["rows"]:
        print(f"  {r['strategy']:45s} reward={r['finding_recall (reward)']:.3f}  words={r['mean_words']:.0f}")


# ---------------------------------------------------------------------------
# imaging_indication
# ---------------------------------------------------------------------------

def imaging_context(row: dict, patients: dict[int, benchmark.Instance]) -> dict:
    inst = patients[row["patient_id"]]
    current = next(e for e in inst.encounters if e["encounter_id"] == row["encounter_id"])
    prior = [e for e in inst.encounters if e["encounter_order"] < current["encounter_order"]]

    def note(e: dict) -> str:
        head = f"ENCOUNTER {e['encounter_date']} ({e['encounter_type']})"
        return head + "\n" + "\n".join(f"[{s['section_type'].upper()}]\n{s['section_text']}" for s in e["sections"])

    by_enc = {}
    for d in inst.gt_diagnoses:
        by_enc.setdefault(d.get("first_encounter_id"), d["display_name"])
    return {"current": current, "prior": prior, "note": note(current),
            "history": "\n\n".join(note(e) for e in prior + [current]),
            "n_prior": len(prior), "diagnosis_name": by_enc.get(row["encounter_id"])}


def submission(question: str) -> dict:
    return {"clinical_question": question, "pre_read_summary": "", "must_include_findings": [], "differential": []}


async def score_imaging(sim: Simulator, rows: list[dict], questions: list[str]) -> list[dict]:
    return await sim.score_batch([(r["gt_id"], submission(q)) for r, q in zip(rows, questions)],
                                 task="imaging_indication")


def summarize_imaging(results: list[dict]) -> dict:
    return {"concept_f1 (reward)": mean(x["reward"] for x in results),
            "concept_precision": mean(x["metrics"].get("clinical_question_concept_precision") or 0 for x in results),
            "concept_recall": mean(x["metrics"].get("clinical_question_concept_recall") or 0 for x in results),
            "token_f1": mean(x["metrics"]["clinical_question_f1"] for x in results)}


async def imaging_strategies() -> None:
    patients = dev_instances()
    rows = task_rows("imaging_indication", patients)
    ctx = [imaging_context(r, patients) for r in rows]
    rng = random.Random(0)
    all_refs = [r["ground_truth"]["inferred_clinical_question"] for r in rows]
    dx_names = [d["display_name"] for d in benchmark.diagnosis_table()]

    strategies = {
        "oracle (reference question)": lambda r, c: r["ground_truth"]["inferred_clinical_question"],
        "empty": lambda r, c: "",
        "order's indication, verbatim": lambda r, c: r["order"]["clinical_indication"],
        "chief complaint of the encounter": lambda r, c: c["current"]["chief_complaint"] or "",
        "chief complaint + indication": lambda r, c: f"Evaluate {r['order']['body_region']} {r['order']['modality']} "
                                                     f"for {r['order']['clinical_indication']} in patient with "
                                                     f"{c['current']['chief_complaint']}",
        "HPI of the encounter, verbatim": lambda r, c: next(
            (s["section_text"] for s in c["current"]["sections"] if s["section_type"] == "hpi"), ""),
        "another patient's reference question": lambda r, c: rng.choice(all_refs),
        "[labels] encounter's reference diagnosis name": lambda r, c: f"Evaluate for {c['diagnosis_name']}"
        if c["diagnosis_name"] else r["order"]["clinical_indication"],
        "[labels] reference differential names": lambda r, c: "Evaluate for " + ", ".join(
            d.get("diagnosis", "") for d in r["ground_truth"].get("differential_context", [])),
        "[labels] reference question + 5 unrelated diagnoses": lambda r, c: r["ground_truth"][
            "inferred_clinical_question"] + ". Also evaluate for " + ", ".join(rng.sample(dx_names, 5)),
    }
    out = {"n": len(rows), "n_patients": len({r["patient_id"] for r in rows}), "rows": []}
    async with Simulator() as sim:
        for name, fn in strategies.items():
            res = await score_imaging(sim, rows, [fn(r, c) for r, c in zip(rows, ctx)])
            out["rows"].append({"strategy": name, **summarize_imaging(res)})
    out["mean_prior_encounters"] = mean(c["n_prior"] for c in ctx)
    out["share_first_encounter"] = mean(c["n_prior"] == 0 for c in ctx)
    write("imaging_strategies_dev", out)
    print(f"imaging_indication, dev, n={out['n']} orders over {out['n_patients']} patients; "
          f"mean prior encounters {out['mean_prior_encounters']:.2f}, "
          f"{100 * out['share_first_encounter']:.0f}% are the first encounter")
    for r in out["rows"]:
        print(f"  {r['strategy']:52s} conceptF1={r['concept_f1 (reward)']:.3f}  P={r['concept_precision']:.3f}  "
              f"R={r['concept_recall']:.3f}  tokenF1={r['token_f1']:.3f}")


QUESTION_PROMPT = """You are a radiologist preparing a pre-read. An imaging study has been ordered with a terse indication. \
State the underlying clinical question the study should answer, in one sentence, naming the specific conditions to evaluate for.

Imaging order: {modality} of {body_region}. Indication as written: "{indication}".
{context}
Answer with the clinical question only."""

CONDITIONS = {
    "order only (no chart)": lambda c: "",
    "order + current encounter note": lambda c: f"\nCurrent encounter:\n{c['note']}\n",
    "order + full chart up to the order": lambda c: f"\nPatient record up to this order:\n{c['history']}\n",
}


async def imaging_llm(model: str, concurrency: int) -> None:
    patients = dev_instances()
    rows = task_rows("imaging_indication", patients)
    ctx = [imaging_context(r, patients) for r in rows]
    llm = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                             max_retries=6, timeout=180)
    sem = asyncio.Semaphore(concurrency)

    async def ask(row: dict, c: dict, build) -> str:
        prompt = QUESTION_PROMPT.format(modality=row["order"]["modality"], body_region=row["order"]["body_region"],
                                        indication=row["order"]["clinical_indication"], context=build(c))
        async with sem:
            resp = await llm.chat.completions.create(model=model, temperature=0, max_tokens=300,
                                                     messages=[{"role": "user", "content": prompt}])
        return (resp.choices[0].message.content or "").strip()

    out = {"model": model, "n": len(rows), "rows": []}
    later = [i for i, c in enumerate(ctx) if c["n_prior"] > 0]
    async with Simulator() as sim:
        for name, build in CONDITIONS.items():
            questions = await asyncio.gather(*(ask(r, c, build) for r, c in zip(rows, ctx)))
            res = await score_imaging(sim, rows, list(questions))
            out["rows"].append({"condition": name, **summarize_imaging(res),
                                "concept_f1, orders with prior encounters": mean(res[i]["reward"] for i in later),
                                "examples": [{"indication": rows[i]["order"]["clinical_indication"],
                                              "question": questions[i],
                                              "reference": rows[i]["ground_truth"]["inferred_clinical_question"],
                                              "reward": res[i]["reward"]} for i in range(3)]})
    await llm.close()
    out["n_with_prior_encounters"] = len(later)
    write(f"imaging_llm_{model.replace('/', '_')}_dev", out)
    print(f"imaging_indication single-turn, {model}, dev, n={len(rows)} ({len(later)} with prior encounters)")
    for r in out["rows"]:
        print(f"  {r['condition']:40s} conceptF1={r['concept_f1 (reward)']:.3f}  P={r['concept_precision']:.3f}  "
              f"R={r['concept_recall']:.3f}  | with-prior subset {r['concept_f1, orders with prior encounters']:.3f}")


async def imaging_agent(model: str, concurrency: int, extra_body: dict | None) -> None:
    patients = dev_instances()
    rows = task_rows("imaging_indication", patients)
    cfg = agent.RolloutConfig(model=model, base_url="https://openrouter.ai/api/v1",
                              api_key_env="OPENROUTER_API_KEY", extra_body=extra_body)
    out_dir = config.RUNS_DIR / f"imaging-{model.replace('/', '_')}-dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = out_dir / "trajectories.jsonl"
    done: dict[int, dict] = {}
    if traj.exists():
        for line in traj.read_text().splitlines():
            rec = json.loads(line)
            if rec["stop_reason"] != "llm_error":
                done[rec["gt_id"]] = rec
    llm = openai.AsyncOpenAI(base_url=cfg.base_url, api_key=os.environ["OPENROUTER_API_KEY"], max_retries=6, timeout=120)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    async with Simulator() as sim:
        async def one(gt_id: int) -> None:
            async with sem:
                rec = await agent.rollout(sim, llm, cfg, gt_id)
            async with lock:
                done[gt_id] = rec
                with traj.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        await asyncio.gather(*(one(r["gt_id"]) for r in rows if r["gt_id"] not in done))
    await llm.close()
    recs = [done[r["gt_id"]] for r in rows if done[r["gt_id"]]["stop_reason"] != "llm_error"]
    tools: dict[str, int] = {}
    for r in recs:
        for t in r["tool_sequence"]:
            tools[t] = tools.get(t, 0) + 1
    out = {"model": model, "n": len(recs), "concept_f1 (reward)": mean(r["reward"] for r in recs),
           "submit_rate": mean(r["submitted"] for r in recs), "valid_rate": mean(r["valid"] for r in recs),
           "budget_exhausted_rate": mean(r["budget_exhausted"] for r in recs),
           "mean_actions": mean(r["actions"] for r in recs),
           "mean_tokens": mean(r["prompt_tokens"] + r["completion_tokens"] for r in recs),
           "concept_precision": mean((r["metrics"] or {}).get("clinical_question_concept_precision") or 0 for r in recs),
           "concept_recall": mean((r["metrics"] or {}).get("clinical_question_concept_recall") or 0 for r in recs),
           "tool_calls": dict(sorted(tools.items(), key=lambda kv: -kv[1]))}
    write(f"imaging_agent_{model.replace('/', '_')}_dev", out)
    print(json.dumps(out, indent=1))


def main() -> None:
    config.load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["retrieval", "summarization", "imaging-strategies", "imaging-llm", "imaging-agent"])
    ap.add_argument("--model", default="qwen/qwen3-30b-a3b-instruct-2507")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--extra-body", default=None)
    args = ap.parse_args()
    if args.command == "retrieval":
        asyncio.run(retrieval())
    elif args.command == "summarization":
        asyncio.run(summarization())
    elif args.command == "imaging-strategies":
        asyncio.run(imaging_strategies())
    elif args.command == "imaging-llm":
        asyncio.run(imaging_llm(args.model, args.concurrency))
    else:
        asyncio.run(imaging_agent(args.model, args.concurrency,
                                  json.loads(args.extra_body) if args.extra_body else None))


if __name__ == "__main__":
    main()
