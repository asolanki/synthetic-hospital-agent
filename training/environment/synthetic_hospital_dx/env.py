"""The paper's agent-loop control flow on top of Verifiers' default single-agent run.

Verifiers' default ends a rollout the first time the model replies without a tool call.
The paper's harness (eval/agents/agent_loop.py, structured arm) does two more things,
ported here so scores stay comparable with the paper and with `baselines/agent.py`:

1. Nudge. A reply with no tool call gets a user message asking for an action, and the
   episode continues, until the turn budget is used.
2. Forced submission. If the model still has not submitted, one last user turn asks for
   the answer now, "by calling the submit tool, or as a single JSON object". A JSON
   answer written as text is parsed and submitted on the model's behalf.

Without these, 47 of 120 dev episodes of Qwen3-30B ended at reward 0 because the model
wrote its answer as JSON text instead of calling `submit_diagnosis`.
"""

from __future__ import annotations

import json
import re

import verifiers.v1 as vf

from synthetic_hospital_dx import simulator
from synthetic_hospital_dx.tools import SUBMIT_TOOL

ANSWER_KEYS = ("active_diagnoses", "chronic_conditions")

# Verbatim from the paper's agent loop.
NUDGE = ("Please take an action. Either call a tool/run a command, or submit your answer. "
         "You have {remaining} actions remaining.")
FORCED = ("You have no actions remaining and have not submitted an answer. Do not call any further "
          "tools. Submit your best answer NOW for the patient_diagnosis task, using the information you "
          "have already gathered. An incomplete answer is better than no answer.\n\n"
          "Your answer must contain exactly these fields:\n"
          "  - active_diagnoses: Active diagnoses (acute or acute_on_chronic)\n"
          "  - chronic_conditions: Chronic conditions\n\n"
          "Return it by calling the submit tool, or as a single JSON object with those fields and nothing else.")


def submission_from_text(text: str) -> dict | None:
    """The largest balanced JSON object in `text` that carries the answer keys (paper's rule,
    without its repair heuristics for truncated JSON)."""
    candidates: list[tuple[int, object]] = []
    for m in re.finditer(r"\{", text):
        depth = 0
        for i in range(m.start(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidates.append((i + 1 - m.start(), json.loads(text[m.start():i + 1])))
                    except json.JSONDecodeError:
                        pass
                    break
    for _, obj in sorted(candidates, key=lambda c: -c[0]):
        payload = obj.get("payload", obj) if isinstance(obj, dict) else obj
        if isinstance(payload, dict) and any(k in payload for k in ANSWER_KEYS):
            return payload
    return None


async def submit_text_answer(trace: vf.Trace, text: str) -> None:
    """Submit an answer the model wrote as text, recording the outcome on the trace state."""
    state = trace.state
    answer = submission_from_text(text)
    if answer is None or not state.episode_id:
        return
    try:
        out = await simulator.post("/env/step", {"episode_id": state.episode_id, "name": SUBMIT_TOOL,
                                                 "arguments": answer})
    except Exception:  # noqa: BLE001 — the upstream scorer raises on some malformed answers
        state.submitted, state.malformed, state.reward, state.from_text = True, True, 0.0, True
        await simulator.post("/env/close", {"episode_id": state.episode_id})
        return
    info = out["info"]
    state.submitted, state.forced, state.from_text = True, True, True
    state.malformed = bool(info.get("malformed"))
    state.reward = float(out["reward"] or 0.0)
    state.actions = out["step"]
    state.metrics = {k: float(v) for k, v in (info.get("metrics") or {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}


class SyntheticHospitalDxEnv(vf.SingleAgentEnv):
    async def run(self, task, agents) -> None:
        budget = task.data.budget
        async with agents.agent.interaction(task) as interaction:
            trace = interaction.trace
            segment = await interaction.turn()
            # 1. Nudge after every tool-less reply while model turns remain (the paper's loop
            #    counts model calls against the same budget as actions).
            while not segment.terminated and not trace.state.submitted and trace.num_turns < budget:
                trace.state.nudges += 1
                remaining = max(budget - trace.state.actions, 0)
                segment = await interaction.turn(NUDGE.format(remaining=remaining))
            # 2. One forced-submission turn; accept a JSON answer written as text.
            if not segment.terminated and not trace.state.submitted:
                segment = await interaction.turn(FORCED)
                if not segment.terminated and not trace.state.submitted:
                    await submit_text_answer(trace, segment.last_reply)
