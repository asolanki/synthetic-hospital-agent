"""Scripted demonstrations for SFT (behaviour cloning of a short, label-informed policy).

For each training patient the demonstration is a real episode on the simulator:

    view_encounters(patient)            -> the visit list (real observation)
    view_encounter_detail(latest visit) -> the latest note, whose problem list carries
                                           every earlier visit's diagnosis (real observation)
    submit_diagnosis(reference answer)  -> scored 1.0 by construction

The observations are exactly what the environment returns, and the tool schemas are the
ones the model sees when evaluated through Verifiers, so SFT and evaluation share one
interface. The submission is the reference answer; the model is taught to read the latest
note and write out the full problem list, including the final visit's diagnosis that the
note does not name. That last part is what makes this "behaviour cloning of an oracle":
the demonstration knows something the observations do not fully reveal.

Output: data/sft/<name>/train.jsonl with prime-rl's `messages` and `tools` columns, plus
meta.json and a copy of the trace the tool schemas were taken from.

    uv run python -m training.sft_data --name oracle-copy-v1 \
        --tools-from data/sft/oracle-copy-v1/tools_source_trace.jsonl
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from common import config, splits
from common.simulator import Simulator


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def _oracle_arguments(oracle: dict) -> dict:
    """The label-derived submission, in the submit tool's schema (it has no 'unspecified')."""
    active = [{**d, "acuity": d["acuity"] if d["acuity"] in ("acute", "acute_on_chronic") else "acute"}
              for d in oracle["active_diagnoses"]]
    chronic = [{**d, "acuity": "chronic"} for d in oracle["chronic_conditions"]]
    return {"active_diagnoses": active, "chronic_conditions": chronic}


async def demonstration(sim: Simulator, gt_id: int, system_prompt: str | None) -> dict:
    ep = await sim.reset(gt_id)
    eid, pid = ep["episode_id"], ep["patient_id"]
    messages = [{"role": "system", "content": system_prompt or ep["instructions"]},
                {"role": "user", "content": ep["intro"]}]

    async def act(call_id: str, name: str, arguments: dict) -> dict:
        out = await sim.step(eid, name, arguments)
        messages.append(_call(call_id, name, arguments))
        messages.append({"role": "tool", "tool_call_id": call_id, "content": out["observation_text"]})
        return out

    visits = (await act("call_0", "view_encounters", {"patient_id": pid}))["observation"]
    latest = max(visits, key=lambda v: (v["date"], v["encounter_id"]))
    await act("call_1", "view_encounter_detail", {"encounter_id": latest["encounter_id"]})
    submission = _oracle_arguments((await sim.oracle(eid))["arguments"])
    final = await act("call_2", "submit_diagnosis", submission)
    return {"gt_id": gt_id, "patient_id": pid, "reward": final["reward"], "messages": messages}


async def build(name: str, tools_from_trace: Path, system_prompt_file: Path | None = None,
                split: str = "train", concurrency: int = 16) -> dict:
    """Write data/sft/<name>/train.jsonl. `tools_from_trace` is any Verifiers traces.jsonl from
    this taskset; its tool list is what the model sees at evaluation time."""
    episode = json.loads(tools_from_trace.read_text().splitlines()[0])
    tools = episode["traces"][0]["tools"]
    # Traces store tools flat ({type, name, description, parameters}); SFT wants chat format.
    tools = [t if "function" in t else {"type": "function", "function": {
        k: t[k] for k in ("name", "description", "parameters") if k in t}} for t in tools]
    system_prompt = system_prompt_file.read_text().strip() if system_prompt_file else None

    gt_ids = splits.gt_ids(split)
    sem = asyncio.Semaphore(concurrency)
    async with Simulator() as sim:
        async def one(gt_id: int) -> dict:
            async with sem:
                return await demonstration(sim, gt_id, system_prompt)
        demos = await asyncio.gather(*(one(g) for g in gt_ids))

    out_dir = config.SFT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train.jsonl").open("w") as f:
        for d in demos:
            f.write(json.dumps({"gt_id": d["gt_id"], "messages": d["messages"], "tools": json.dumps(tools)}) + "\n")
    copied = out_dir / "tools_source_trace.jsonl"
    if tools_from_trace.resolve() != copied.resolve():
        copied.write_text(tools_from_trace.read_text())
    rewards = [d["reward"] for d in demos]
    meta = {"name": name, "split": split, "n": len(demos), "mean_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards), "system_prompt": str(system_prompt_file) if system_prompt_file else "paper",
            "policy": "view_encounters -> view_encounter_detail(latest) -> submit(reference)",
            "tools_from": str(copied.relative_to(config.ROOT))}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Write scripted SFT demonstrations for the training patients.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tools-from", required=True, type=Path,
                    help="a Verifiers traces.jsonl from synthetic-hospital-dx (source of the tool schemas)")
    ap.add_argument("--system-prompt", type=Path, default=None, help="prompt file; default is the paper prompt")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(build(args.name, args.tools_from, args.system_prompt)), indent=1))


if __name__ == "__main__":
    main()
