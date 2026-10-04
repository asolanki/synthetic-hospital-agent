"""How well can a model code the diagnosis names that the chart already lists?

The audit's copy strategy maps each dated "Active Problem List" entry to a code
with the benchmark's own name-to-code table, which no model can see. This script
replaces that table with a model: it is shown only the copied names (no chart)
and asked for ICD-10-CM codes. The resulting score is the realistic ceiling of
"copy the dated entries and code them yourself", and the gap to the lookup
version measures how much of the task is ICD-10 recall (and label noise where
the reference code does not match the reference name).

    uv run python -m audit.coding_ceiling --model qwen/qwen3-30b-a3b-instruct-2507
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

import openai

from audit import strategies
from common import benchmark, config, metrics, splits
from common.simulator import Simulator

PROMPT = """Give the ICD-10-CM code for each diagnosis below. Use the code a professional coder would assign \
to exactly that condition. Answer with a JSON object mapping each diagnosis, verbatim, to its code, and nothing else.

{names}"""


async def code_names(llm, model: str, names: list[str], sem: asyncio.Semaphore) -> dict[str, str]:
    if not names:
        return {}
    async with sem:
        resp = await llm.chat.completions.create(
            model=model, temperature=0, max_tokens=1500,
            messages=[{"role": "user", "content": PROMPT.format(names="\n".join(f"- {n}" for n in names))}])
    text = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, flags=re.S)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except ValueError:
        parsed = {}
    return {n: str(parsed.get(n) or "") for n in names}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    config.load_dotenv()

    by_id = {i.gt_id: i for i in benchmark.load_instances(("train",))}
    insts = [by_id[g] for g in splits.gt_ids(args.split)]
    llm = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                             max_retries=6, timeout=120)
    sem = asyncio.Semaphore(args.concurrency)
    dated = [[n for n, is_dated in strategies._problem_list_lines(i) if is_dated] for i in insts]
    coded = await asyncio.gather(*(code_names(llm, args.model, names, sem) for names in dated))
    await llm.close()

    preds = [strategies._submission([(codes[n], n) for n in names]) for names, codes in zip(dated, coded)]
    async with Simulator() as sim:
        results = await sim.score_batch([(i.gt_id, p) for i, p in zip(insts, preds)])
    records = [{"gt_id": i.gt_id, "n_pred": metrics.count_predictions(p), "reward": r["reward"], "metrics": r["metrics"]}
               for i, p, r in zip(insts, preds, results)]

    table = benchmark.name_to_icd()
    pairs = [(n, codes[n], table.get(n.strip().lower())) for names, codes in zip(dated, coded) for n in names]
    known = [(n, c, ref) for n, c, ref in pairs if ref]
    agree = sum(metrics.match_key(c) == metrics.match_key(ref) for _, c, ref in known)
    out = {
        "model": args.model, "split": args.split, **metrics.summarize(records, n_boot=300),
        "visit_split": metrics.split_recall(list(zip(insts, preds))),
        "names_coded": len(pairs), "names_with_reference_code": len(known),
        "category_agreement_with_reference": agree / len(known) if known else None,
        "disagreements": [{"name": n, "model": c, "reference": ref} for n, c, ref in known
                          if metrics.match_key(c) != metrics.match_key(ref)],
    }
    config.write_result(f"audit/coding_ceiling_{args.model.replace('/', '_')}_{args.split}", out)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))
    for d in out["disagreements"][:25]:
        print(f"  {d['name']!r}: model {d['model']} vs reference {d['reference']}")


if __name__ == "__main__":
    asyncio.run(main())
