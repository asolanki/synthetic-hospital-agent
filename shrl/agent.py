"""Tool-use rollouts of an OpenAI-compatible chat model in the /env environment.

One episode = one patient. The model receives the environment's own system
prompt, patient intro, and function schemas (the paper's agent prompt, unless
`system_prompt` overrides it), acts through native function calling, and ends
the episode with `submit_diagnosis`. Every episode is logged in full so the
same files serve evaluation, behaviour analysis, and later SFT data.

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

from shrl import config, metrics
from shrl.client import Simulator, SimulatorError

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
