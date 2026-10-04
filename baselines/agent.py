"""The first agent harness: tool-use episodes of a chat model against the simulator.

Built for milestone 1 to get baseline numbers through OpenRouter before the
Verifiers environment existed (training/environment/). It plays the same episode
(same server, prompt, tools, budget, nudge rule) and scores the same on the
120 dev patients (30B: 0.463 here, 0.466 there), but prime-rl cannot use it.
New runs go through training/; this stays for reproducing milestone-1 tables.

    uv run python -m baselines.agent --model qwen/qwen3-30b-a3b-instruct-2507 --split dev
    uv run python -m baselines.agent --model ... --prompt-file baselines/prompts/strong_v1.md

Full trajectories go to runs/<name>/, the summary to results/raw/baselines/<name>.json.

One episode = one patient. The model receives the environment's own system
prompt, patient intro, and function schemas (the paper's agent prompt, unless
`system_prompt` overrides it), acts through native function calling, and ends
the episode with `submit_diagnosis`.

Episode outcome fields:
  submitted         the submit tool was called (the environment ended the episode)
  valid             submitted, well-formed, and scored (non-empty)
  budget_exhausted  the action budget was spent before the episode ended
  stop_reason       submitted | no_tool_call | max_turns | context_overflow | llm_error
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import openai

from common import config, metrics
from common.simulator import Simulator, SimulatorError

NUDGE = ("You did not call a tool. Continue by calling one of the available tools, or call "
         "{submit_tool} to submit your final answer.")


@dataclass
class RolloutConfig:
    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2048
    budget: int = config.DEFAULT_BUDGET
    max_no_tool_turns: int = 2
    seed: int | None = 0
    system_prompt: str | None = None      # None = the environment's (paper) prompt
    extra_body: dict | None = None        # provider-specific request fields


def _assistant_message(msg) -> dict:
    out: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [{"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                             for tc in msg.tool_calls]
    return out


async def rollout(sim: Simulator, llm: openai.AsyncOpenAI, cfg: RolloutConfig, gt_id: int) -> dict:
    started = time.time()
    ep = await sim.reset(gt_id, budget=cfg.budget)
    episode_id = ep["episode_id"]
    messages: list[dict] = [
        {"role": "system", "content": cfg.system_prompt or ep["instructions"]},
        {"role": "user", "content": ep["intro"]},
    ]
    record = {
        "gt_id": gt_id, "patient_id": ep["patient_id"], "model": cfg.model, "budget": cfg.budget,
        "submitted": False, "valid": False, "malformed": False, "budget_exhausted": False,
        "stop_reason": None, "reward": 0.0, "metrics": None, "n_pred": 0,
        "actions": 0, "turns": 0, "tool_errors": 0, "bad_json_calls": 0, "no_tool_turns": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "peak_context_tokens": 0,
        "tool_sequence": [], "submission": None, "error": None,
    }
    no_tool_streak = 0

    for _ in range(cfg.budget + 10):
        try:
            resp = await llm.chat.completions.create(
                model=cfg.model, messages=messages, tools=ep["tools"], tool_choice="auto",
                temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
                seed=cfg.seed, extra_body=cfg.extra_body)
        except openai.BadRequestError as exc:
            overflow = any(s in str(exc).lower() for s in ("context", "maximum", "too long", "tokens"))
            record.update(stop_reason="context_overflow" if overflow else "llm_error", error=str(exc)[:500])
            break
        except openai.OpenAIError as exc:
            record.update(stop_reason="llm_error", error=str(exc)[:500])
            break

        record["turns"] += 1
        if resp.usage:
            record["prompt_tokens"] += resp.usage.prompt_tokens
            record["completion_tokens"] += resp.usage.completion_tokens
            record["peak_context_tokens"] = max(record["peak_context_tokens"],
                                                resp.usage.prompt_tokens + resp.usage.completion_tokens)
        msg = resp.choices[0].message
        messages.append(_assistant_message(msg))

        if not msg.tool_calls:
            record["no_tool_turns"] += 1
            no_tool_streak += 1
            if no_tool_streak > cfg.max_no_tool_turns:
                record["stop_reason"] = "no_tool_call"
                break
            messages.append({"role": "user", "content": NUDGE.format(submit_tool=ep["submit_tool"])})
            continue
        no_tool_streak = 0

        done = False
        for tc in msg.tool_calls:
            name = tc.function.name
            if done:   # calls issued after the submission in the same turn are not executed
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Episode already finished."})
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                # Not forwarded: an unparseable call would otherwise end the episode if it was the submit tool.
                record["bad_json_calls"] += 1
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"error": f"tool arguments were not valid JSON: {exc}"})})
                continue
            try:
                out = await sim.step(episode_id, name, args)
            except SimulatorError as exc:
                if name != ep["submit_tool"]:
                    raise
                # The upstream scorer raises on some malformed submissions (e.g. a list of
                # strings) instead of scoring 0. Treat it as what it is: a malformed submission.
                state = await sim.close(episode_id)
                record.update(submitted=True, malformed=True, reward=0.0, submission=args,
                              actions=state["steps"] + 1, stop_reason="submitted",
                              error=f"scorer error on submission: {str(exc)[:300]}")
                record["tool_sequence"].append(name)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"status": "submitted", "malformed": True})})
                done = True
                continue
            record["actions"] = out["step"]
            record["tool_sequence"].append(name)
            if isinstance(out["observation"], dict) and out["observation"].get("error"):
                record["tool_errors"] += 1
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out["observation_text"]})
            if out["remaining"] <= 0 and not out["done"]:
                record["budget_exhausted"] = True
            if out["done"]:
                done = True
                info = out["info"]
                record.update(
                    submitted=True, malformed=bool(info.get("malformed")), reward=out["reward"],
                    metrics=info.get("metrics"), submission=args, n_pred=metrics.count_predictions(args),
                    budget_exhausted=record["budget_exhausted"] or bool(info.get("forced")),
                    stop_reason="submitted")
        if done:
            break
    else:
        record["stop_reason"] = "max_turns"

    if not record["submitted"]:
        await sim.close(episode_id)
    if record["submitted"] and record["malformed"] and record["error"]:
        record["budget_exhausted"] = record["budget_exhausted"] or record["actions"] > cfg.budget
    # patient_diagnosis reports n_scored (0 for an empty list); other tasks have no such field.
    record["valid"] = bool(record["submitted"] and not record["malformed"] and record["metrics"]
                           and record["metrics"].get("n_scored", 1) == 1)
    record["seconds"] = round(time.time() - started, 2)
    record["messages"] = messages
    return record


def summarize_run(records: list[dict]) -> dict:
    n = len(records)
    mean = lambda key: sum(r[key] for r in records) / n if n else 0.0  # noqa: E731
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["stop_reason"]] = reasons.get(r["stop_reason"], 0) + 1
    return {
        **metrics.summarize(records),
        "submit_rate": mean("submitted"),
        "budget_exhausted_rate": mean("budget_exhausted"),
        "mean_actions": mean("actions"),
        "mean_turns": mean("turns"),
        "mean_tool_errors": mean("tool_errors"),
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_completion_tokens": mean("completion_tokens"),
        "mean_total_tokens": mean("prompt_tokens") + mean("completion_tokens"),
        "mean_peak_context_tokens": mean("peak_context_tokens"),
        "stop_reasons": reasons,
    }


async def run(cfg: RolloutConfig, gt_ids: list[int], out_dir: Path, concurrency: int = 8) -> dict:
    """Roll out every gt_id once, appending to out_dir/trajectories.jsonl (resumable)."""
    import os

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    records: list[dict] = []
    if traj_path.exists():
        # Episodes that died on a provider error (rate limit, outage) say nothing about the
        # model; they are retried on resume and the latest attempt per instance wins.
        latest = {}
        for line in traj_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[rec["gt_id"]] = rec
        records = [r for r in latest.values() if r["stop_reason"] != "llm_error"]
    done_ids = {r["gt_id"] for r in records}
    todo = [g for g in gt_ids if g not in done_ids]

    llm = openai.AsyncOpenAI(base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env) or "EMPTY",
                             max_retries=8, timeout=120)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async with Simulator() as sim:
        async def one(gt_id: int) -> None:
            async with sem:
                rec = await rollout(sim, llm, cfg, gt_id)
            async with lock:
                records.append(rec)
                with traj_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{len(records)}/{len(gt_ids)}] gt_id={gt_id} reward={rec['reward']:.3f} "
                      f"actions={rec['actions']} stop={rec['stop_reason']}", flush=True)

        await asyncio.gather(*(one(g) for g in todo))
    await llm.close()

    wanted = set(gt_ids)
    records = [r for r in records if r["gt_id"] in wanted]
    n_provider_errors = sum(r["stop_reason"] == "llm_error" for r in records)
    records = [r for r in records if r["stop_reason"] != "llm_error"]
    summary = {"config": asdict(cfg), "n_requested": len(gt_ids),
               "n_excluded_provider_errors": n_provider_errors, **summarize_run(records)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> None:
    import argparse

    from baselines import report
    from common import splits

    config.load_dotenv()
    ap = argparse.ArgumentParser(description="Roll a chat model out on the dev patients (early harness).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--split", choices=["dev", "train", "public"], default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--budget", type=int, default=config.DEFAULT_BUDGET)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-body", default=None, help="JSON merged into the chat request body")
    ap.add_argument("--prompt-file", default=None,
                    help="system prompt file relative to the repo root (default: the environment's paper prompt)")
    args = ap.parse_args()

    gt_ids = splits.gt_ids(args.split)
    if args.limit:
        gt_ids = gt_ids[:args.limit]
    cfg = RolloutConfig(
        model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
        budget=args.budget, seed=args.seed,
        system_prompt=(config.ROOT / args.prompt_file).read_text().strip() if args.prompt_file else None,
        extra_body=json.loads(args.extra_body) if args.extra_body else None)
    name = args.name or f"{args.model.replace('/', '_').replace(':', '_')}-{args.split}"
    summary = asyncio.run(run(cfg, gt_ids, config.RUNS_DIR / name, concurrency=args.concurrency))
    summary.update(name=name, split=args.split)
    config.write_result(f"baselines/{name}", summary)
    print(report.baseline_table([summary]))


if __name__ == "__main__":
    main()
