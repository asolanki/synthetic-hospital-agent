"""Measure complete hospital rollouts at several concurrency levels on one vLLM server.

    uv run python -m training.throughput --tasks 64 --concurrency 8 16 32 64

Each trial uses the same dev patients and sampling seed, but a different vLLM
cache salt: no cross-trial cache reuse, while normal within-trial prefix reuse
remains enabled. The server must have no other inference clients during the run.
Raw traces, command output, and two-second resource samples are retained.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tomllib
from uuid import uuid4

import httpx

from common import config

METRICS = (
    "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
    "num_preemptions_total", "prompt_tokens_total", "generation_tokens_total",
    "prefix_cache_queries_total", "prefix_cache_hits_total",
)


def engine_metrics(client: httpx.Client) -> dict[str, float]:
    response = client.get("/metrics")
    response.raise_for_status()
    out = {}
    for line in response.text.splitlines():
        if line.startswith("#") or not line:
            continue
        label, value = line.rsplit(" ", 1)
        name = label.split("{", 1)[0].removeprefix("vllm:")
        if name in METRICS:
            out[name] = out.get(name, 0.0) + float(value)
    missing = set(METRICS) - out.keys()
    if missing:
        raise ValueError(f"vLLM metrics missing: {sorted(missing)}")
    return out


def run_trial(base: dict, directory: Path, tasks: int, concurrency: int) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    trial = {**base, "max_concurrent": concurrency, "select": {"limit": tasks},
             "output_dir": str(directory), "run": {"name": directory.name, "dir": "eval"},
             "sampling": {**base["sampling"], "cache_salt": uuid4().hex}}
    settings = directory / "config.json"
    settings.write_text(json.dumps(trial, indent=2) + "\n")
    command = [str(Path(sys.executable).with_name("vf-eval")), "@", str(settings), "--no-rich"]
    base_url = base["client"]["base_url"].removesuffix("/v1")
    samples = []
    with httpx.Client(base_url=base_url, timeout=10) as client:
        before = engine_metrics(client)
        if before["num_requests_running"] or before["num_requests_waiting"]:
            raise RuntimeError("The inference server has other requests; benchmark needs an idle server")
        start = time.monotonic()
        with (directory / "eval.log").open("w") as log, (directory / "samples.jsonl").open("w") as measurements:
            process = subprocess.Popen(command, cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    sample = {"elapsed_seconds": time.monotonic() - start, **engine_metrics(client)}
                    gpu = subprocess.run([
                        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ], check=True, capture_output=True, text=True, timeout=10)
                    util, memory, watts = map(float, gpu.stdout.strip().split(","))
                    sample.update(gpu_util_percent=util, gpu_memory_mib=memory, gpu_power_watts=watts)
                    ping_start = time.monotonic()
                    health = httpx.get("http://127.0.0.1:58000/health", timeout=10)
                    health.raise_for_status()
                    sample["hospital_health_ms"] = (time.monotonic() - ping_start) * 1000
                    samples.append(sample)
                    measurements.write(json.dumps(sample) + "\n")
                    measurements.flush()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
            elapsed = time.monotonic() - start
            after = engine_metrics(client)
        if process.returncode:
            raise RuntimeError(f"vf-eval exited {process.returncode}; see {directory / 'eval.log'}")
    episodes = [json.loads(line) for line in (directory / "eval/traces.jsonl").read_text().splitlines()]
    if len(episodes) != tasks:
        raise RuntimeError(f"Incomplete trial: {len(episodes)}/{tasks}; inspect saved traces")
    successful = sum(e["ok"] and e["traces"][0]["metrics"].get("submitted", 0) for e in episodes)
    output_tokens = sum(e["num_output_tokens"] for e in episodes)
    result = {
        "tasks": tasks, "concurrency": concurrency, "seconds": elapsed,
        "episodes_per_hour": tasks * 3600 / elapsed,
        "successful_episodes_per_hour": successful * 3600 / elapsed,
        "rollout_error_rate": statistics.mean(not e["ok"] for e in episodes),
        "model_call_error_episode_rate": statistics.mean(
            any(call.get("error") for call in e["traces"][0]["calls"]) for e in episodes
        ),
        "model_call_error_types": dict(Counter(
            call["error"]["type"] for e in episodes for call in e["traces"][0]["calls"] if call.get("error")
        )),
        "output_tokens": output_tokens, "output_tokens_per_second": output_tokens / elapsed,
        "mean_reward": statistics.mean(
            e["traces"][0]["rewards"]["official_score"]["score"] if e["ok"]
            else e["traces"][0]["rewards"].get("official_score", {}).get("score", 0.0)
            for e in episodes
        ),
        "submitted_rate": statistics.mean(e["traces"][0]["metrics"].get("submitted", 0) for e in episodes),
        "gpu_util_mean_percent": statistics.mean(s["gpu_util_percent"] for s in samples),
        "gpu_util_max_percent": max(s["gpu_util_percent"] for s in samples),
        "kv_cache_peak_fraction": max(s["kv_cache_usage_perc"] for s in samples),
        "requests_running_peak": max(s["num_requests_running"] for s in samples),
        "requests_waiting_peak": max(s["num_requests_waiting"] for s in samples),
        "hospital_health_mean_ms": statistics.mean(s["hospital_health_ms"] for s in samples),
        "preemptions": after["num_preemptions_total"] - before["num_preemptions_total"],
        "prefix_cache_hits": after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"],
        "prefix_cache_queries": after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"],
        "trace_file": str(directory / "eval/traces.jsonl"),
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/configs/eval_local.toml"))
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/a100-throughput"))
    args = parser.parse_args()
    if args.tasks < max(args.concurrency) or min(args.concurrency) < 1:
        parser.error("tasks must cover the largest concurrency; concurrency must be positive")
    base = tomllib.loads(args.config.read_text())
    results = []
    for concurrency in args.concurrency:
        result = run_trial(base, args.output_dir.resolve() / f"c{concurrency}", args.tasks, concurrency)
        results.append(result)
        (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
