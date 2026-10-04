"""Synthetic Hospital `patient_diagnosis`, played as the paper's tool-use task.

One task = one patient. Each rollout opens a fresh episode on the simulator's
corrected `/env` route (server-side filtering, action budget, scoring), exposes the
paper's EHR tools to the model over MCP, and ends when the model calls
`submit_diagnosis`. The reward is the official per-episode score
(`weighted_problem_list_f1_neutral`), computed by the simulator; labels never leave it.

    uv run vf-eval @ training/configs/eval_openrouter.toml -n 5

The default system prompt is the paper's, fetched from the simulator; override it
with `--env.taskset.system-prompt baselines/prompts/strong_v1.md`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import verifiers.v1 as vf

from synthetic_hospital_dx import simulator
from synthetic_hospital_dx.state import EpisodeState
from synthetic_hospital_dx.tools import EHRToolset

TASK = "patient_diagnosis"
# training/environment/synthetic_hospital_dx/taskset.py -> repo root is parents[3].
DEFAULT_SPLIT_FILE = Path(__file__).resolve().parents[3] / "data" / "splits" / "patient_diagnosis.json"


class DxData(vf.TaskData):
    gt_id: int
    patient_id: int
    budget: int


class DxTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class DxTask(vf.Task[DxData, EpisodeState, DxTaskConfig]):
    @property
    def key(self) -> str:
        return f"gt-{self.data.gt_id}"

    @classmethod
    def toolsets(cls, config: DxTaskConfig) -> list[vf.Toolset]:
        return [EHRToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        episode = await simulator.post("/env/reset", {"gt_id": self.data.gt_id, "budget": self.data.budget})
        trace.state.episode_id = episode["episode_id"]

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # Runs after the env's control flow (env.py), which may already have submitted a
        # text answer on the model's behalf.
        # An episode the model never submitted is closed at reward 0, as in the paper.
        if trace.state.episode_id and not trace.state.submitted:
            await simulator.post("/env/close", {"episode_id": trace.state.episode_id})

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        return trace.state.submitted

    @vf.reward
    async def official_score(self, trace: vf.Trace) -> float:
        return trace.state.reward or 0.0

    @vf.metric
    async def episode(self, trace: vf.Trace) -> dict[str, float]:
        s = trace.state
        out = {"submitted": float(s.submitted), "forced_submit": float(s.forced),
               "malformed_submit": float(s.malformed), "submitted_as_text": float(s.from_text),
               "nudges": float(s.nudges), "actions": float(s.actions),
               "tool_errors": float(s.tool_errors)}
        for key in ("problem_list_precision_neutral", "weighted_problem_list_recall", "problem_list_recall",
                    "acuity_accuracy", "n_scored"):
            if key in s.metrics:
                out[key] = s.metrics[key]
        return out


class DxConfig(vf.TasksetConfig):
    split: str = "dev"
    """`train` or `dev` (the fixed split of upstream `train`), or `public`. `heldout` is sealed."""
    split_file: str = os.environ.get("SHRL_SPLIT_FILE", str(DEFAULT_SPLIT_FILE))
    budget: int = 40
    task: DxTaskConfig = DxTaskConfig()


class SyntheticHospitalDxTaskset(vf.Taskset[DxTask, DxConfig]):
    def load(self) -> list[DxTask]:
        split = self.config.split
        if split in ("train", "dev"):
            gt_ids = json.loads(Path(self.config.split_file).read_text())[split]
            source_split = "train"
        elif split == "public":
            gt_ids, source_split = None, "public"
        else:
            raise ValueError(f"split must be train, dev or public, not {split!r} (heldout is sealed)")

        with simulator_sync() as client:
            rows = client.get("/score/instances", params={"task": TASK, "split": source_split, "limit": 5000}).json()
        patient_of = {r["gt_id"]: r["patient_id"] for r in rows}
        if gt_ids is None:
            gt_ids = sorted(patient_of)

        # The paper's system prompt is built by the simulator; it is identical for every
        # patient_diagnosis episode with the same budget, so read it once.
        probe = simulator.post_sync("/env/reset", {"gt_id": gt_ids[0], "budget": self.config.budget})
        simulator.post_sync("/env/close", {"episode_id": probe["episode_id"]})
        system_prompt = probe["instructions"]

        return [
            DxTask(
                DxData(
                    id=str(gt_id), name=f"patient-{patient_of[gt_id]}",
                    gt_id=gt_id, patient_id=patient_of[gt_id], budget=self.config.budget,
                    system_prompt=system_prompt,
                    # Verbatim from eval/agents/prompts.py build_patient_intro.
                    prompt=f"Your assigned patient is patient_id = {patient_of[gt_id]}.\nBegin your investigation now.",
                ),
                self.config.task,
            )
            for gt_id in gt_ids
        ]


def simulator_sync():
    import httpx
    return httpx.Client(base_url=simulator.base_url(), headers=simulator.headers(), timeout=60)
