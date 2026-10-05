"""Publish Prime-RL's native episode stream into its existing W&B run.

Run with the Prime-RL Python environment from the repository root. With --pid,
follow that local launcher until it exits; without it, backfill a completed run.
Curves and training orchestration remain owned by Prime-RL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import wandb
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk

from training.tracking import RolloutLog, tracking_settings


def read_appended(stream: Path, offsets: dict[int, int]):
    """Read complete records across live and sealed chunks, retaining byte offsets."""
    for number in sorted(chunk_numbers(stream)):
        with open_chunk(stream, number) as source:
            source.seek(offsets.get(number, 0))
            while line := source.readline():
                if not line.endswith(b"\n"):
                    break
                yield json.loads(line)
                offsets[number] = source.tell()


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int, help="Local Prime-RL launcher PID to follow")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    stream = get_trace_stream(args.run_dir)
    offsets: dict[int, int] = {}
    run = None
    log = None
    complete = False
    backfill_exit_code = None
    destination = tracking_settings()
    try:
        while True:
            alive = process_alive(args.pid)
            for episode in read_appended(stream, offsets):
                identity = episode["run"]
                if identity["type"] != "train":
                    raise ValueError("Expected native training-run episodes")
                if run is None:
                    if args.mode == "online" and args.pid is None:
                        native = wandb.Api().run(
                            f"{destination['entity']}/{destination['project']}/{identity['id']}"
                        )
                        if native.summary.get("rollout_tracking_complete"):
                            print(json.dumps({"already_published": True, "run_id": native.id}),
                                  flush=True)
                            return
                        if native.state not in {"finished", "failed"}:
                            raise ValueError(
                                f"Backfill requires a finished/failed run, not {native.state}; "
                                "use --pid for live tracking"
                            )
                        backfill_exit_code = 0 if native.state == "finished" else 1
                    initial = args.run_dir / "initial_adapter"
                    startup = args.run_dir / "broadcasts/step_0"
                    if not initial.exists() and (startup / ".finished").is_file():
                        shutil.copytree(startup, initial)
                    os.environ.setdefault("WANDB_DISABLE_WEAVE", "1")
                    sharing = ({"mode": "shared", "x_label": "hospital-rollouts",
                                "x_primary": False,
                                "x_update_finish_state": backfill_exit_code is not None}
                               if args.mode == "online" else {"mode": "offline"})
                    settings = wandb.Settings(disable_git=True, save_code=False, **sharing)
                    run = wandb.init(
                        **destination, id=identity["id"],
                        name=identity.get("name"), dir=str(args.run_dir), settings=settings,
                    )
                    log = RolloutLog(run)
                    run.summary["rollout_table_scope"] = (
                        "All finalized episodes, including errors and episodes excluded from "
                        "updates. Native effective-batch curves describe the training batches."
                    )
                    run.summary["rollout_tracking_complete"] = False
                    run.summary["initial_adapter_saved"] = (
                        args.run_dir / "initial_adapter/adapter_model.safetensors"
                    ).is_file()
                    print(json.dumps({"wandb_url": run.url, "run_id": run.id}), flush=True)
                if identity["id"] != run.id:
                    raise ValueError("Trace stream contains multiple training runs")
                log.add_episode(episode)
            if log is not None:
                log.flush()
            if not alive:
                break
            time.sleep(5)
        if run is None:
            raise ValueError("No completed episodes found in the native trace stream")
        artifact = wandb.Artifact(f"{run.id}-rollouts", type="rl-rollouts")
        artifact.add_dir(str(stream.parent), name="traces")
        run.log_artifact(artifact)
        run.summary["recorded_episodes"] = log.count
        run.summary["rollout_tracking_complete"] = True
        complete = True
        print(json.dumps({"recorded_episodes": log.count,
                          "all_finalized_mean_reward": log.reward_sum / log.count}), flush=True)
    finally:
        if run is not None:
            # A backfill reopens the SDK run; restore its observed terminal outcome.
            # A live observer leaves finalization to the native training process.
            run.finish(exit_code=(backfill_exit_code if backfill_exit_code is not None
                                  else (0 if complete else 1)))


if __name__ == "__main__":
    main()
