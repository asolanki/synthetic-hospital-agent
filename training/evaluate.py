"""Evaluate all fixed dev patients with the paper prompt and live W&B tracking.

    uv run --group verifiers --group tracking python -m training.evaluate \
        --model shrl-qwen3-4b-sft-oracle-copy-v1 \
        --name qwen3-4b-sft-paper-dev --output-dir runs/sft-dev/paper

`--split train` evaluates the 680 training patients instead, to measure how well a
checkpoint fits the patients it was trained on; it is not a generalization result.
`--split public` evaluates the paper's 200-patient reported split, which our
train/dev partition never touches. Heldout stays sealed.
Uses the existing evaluator, resource sampler and official reporting path.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait
import json
from pathlib import Path
import tomllib

import httpx
import wandb

from common import benchmark, config
from training.report import report
from training.throughput import run_trial
from training.tracking import RolloutLog, publish_report, tracking_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=config.ROOT / "training/configs/eval_local.toml")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--seed", type=int, help="Sampling seed for a replicate; defaults to the config's seed")
    parser.add_argument("--split", choices=("dev", "train", "public"), default="dev",
                        help="Fixed split to evaluate; heldout stays sealed")
    parser.add_argument("--temperature", type=float,
                        help="Decoding diagnostic only: override the paper's temperature 0.7 (0 = greedy, top-p 1)")
    parser.add_argument("--top-p", type=float,
                        help="Decoding diagnostic only: override top-p (e.g. 1.0 with temperature 1 = RL rollout sampling)")
    args = parser.parse_args()
    base = tomllib.loads(args.config.read_text())
    taskset = base["env"]["taskset"]
    if taskset["id"] != "synthetic-hospital-dx" or taskset["split"] != "dev":
        parser.error("Expected the fixed hospital dev evaluation config")
    taskset["split"] = args.split
    taskset.pop("system_prompt", None)
    base["model"] = args.model
    if args.seed is not None:
        base["sampling"]["seed"] = args.seed
    override = None
    if args.temperature is not None or args.top_p is not None:
        temperature = base["sampling"]["temperature"] if args.temperature is None else args.temperature
        top_p = args.top_p if args.top_p is not None else (1.0 if temperature == 0 else base["sampling"]["top_p"])
        override = {"temperature": temperature, "top_p": top_p}
        base["sampling"].update(override)
    allowed = ([i.gt_id for i in benchmark.load_instances(("public",))] if args.split == "public"
               else json.loads(config.SPLIT_FILE.read_text())[args.split])
    directory = args.output_dir.resolve()
    if directory.exists():
        parser.error("Evaluation output directory already exists; never resume or selectively retry")
    directory.parent.mkdir(parents=True, exist_ok=True)
    version = httpx.get(base["client"]["base_url"].removesuffix("/v1") + "/version", timeout=10)
    version.raise_for_status()
    with wandb.init(
        **tracking_settings(), name=args.name, group="qwen3-4b", job_type="evaluation",
        config={**base, "prompt_variant": "paper", "inference_version": version.json()["version"]},
        dir=str(directory.parent), mode=args.mode,
        settings=wandb.Settings(disable_git=True, save_code=False),
    ) as run:
        traces = directory / "eval/traces.jsonl"
        log = RolloutLog(run, traces)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_trial, base, directory, len(allowed), base["max_concurrent"])
            while not future.done():
                log.update()
                wait((future,), timeout=2)
            performance = future.result()
        log.update()
        result = asyncio.run(report(traces, args.split))
        result["summary"]["name"] = args.name
        result.update(performance=performance, inference_version=version.json()["version"],
                      sampling_seed=base["sampling"]["seed"], decoding_override=override, wandb_url=run.url)
        target = config.write_result(f"training/{args.name}", result)
        publish_report(run, target, traces, directory / "config.json")
        print(json.dumps({"wandb_url": run.url, "report": str(target), "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
