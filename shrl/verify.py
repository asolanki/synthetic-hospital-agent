"""Phase 1 benchmark checks: scorer sanity and label exposure in policy-visible observations.

`run()` returns a dict of findings and a list of failed hard checks:

- the oracle submission scores 1.0 on every train and public instance;
- malformed and empty submissions score 0 without raising;
- an episode ends at reward 0 when closed without submission, and the budget is enforced;
- observations from every EHR tool contain no assessment/plan sections and none
  of the reference ICD-10 codes;
- (reported, not a pass/fail) how often reference diagnosis *names* are readable
  in the chart, split by whether they are carried forward into a later
  encounter's "Active Problem List" or only inferable from their own encounter.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import subprocess

from shrl import config, data
from shrl.client import Simulator, SimulatorError

MALFORMED = {
    "empty_object": {},
    "empty_lists": {"active_diagnoses": [], "chronic_conditions": []},
    "wrong_keys": {"diagnosis": "I10"},
    "string_lists": {"active_diagnoses": "I10", "chronic_conditions": "E11"},
    "list_of_strings": {"active_diagnoses": ["I10"], "chronic_conditions": []},
    "null_codes": {"active_diagnoses": [{"icd10": None, "name": "x", "acuity": "acute"}], "chronic_conditions": []},
    "not_icd": {"active_diagnoses": [{"icd10": "hypertension", "name": "x", "acuity": "acute"}], "chronic_conditions": []},
}


async def _oracle_all(sim: Simulator, instances: list[data.Instance], concurrency: int = 16) -> dict:
    sem = asyncio.Semaphore(concurrency)
    bad: list[dict] = []

    async def one(inst: data.Instance) -> float:
        async with sem:
            ep = await sim.reset(inst.gt_id)
            oracle = await sim.oracle(ep["episode_id"])
            out = await sim.step(ep["episode_id"], config.SUBMIT_TOOL, oracle["arguments"])
            if out["reward"] < 1.0:
                bad.append({"gt_id": inst.gt_id, "reward": out["reward"],
                            "n_reference": len(inst.gt_diagnoses)})
            return out["reward"]

    rewards = await asyncio.gather(*(one(i) for i in instances))
    return {"n": len(rewards), "mean_reward": sum(rewards) / len(rewards),
            "n_below_1": len(bad), "below_1": bad[:20]}


async def _malformed(sim: Simulator, gt_id: int) -> dict:
    out = {}
    for name, payload in MALFORMED.items():
        try:
            out[name] = (await sim.score(gt_id, payload))["reward"]
        except SimulatorError as exc:
            out[name] = f"ERROR {exc}"
    # Through the environment as well: a malformed submit must end the episode at 0.
    ep = await sim.reset(gt_id)
    step = await sim.step(ep["episode_id"], config.SUBMIT_TOOL, {"diagnosis": "I10"})
    out["env_malformed_submit"] = {"reward": step["reward"], "done": step["done"],
                                   "malformed": step["info"].get("malformed")}
    # A submission the scorer cannot iterate: does the episode end, or is it left open?
    ep = await sim.reset(gt_id)
    try:
        step = await sim.step(ep["episode_id"], config.SUBMIT_TOOL, MALFORMED["list_of_strings"])
        out["env_list_of_strings_submit"] = {"reward": step["reward"], "done": step["done"]}
    except SimulatorError as exc:
        state = await sim.close(ep["episode_id"])
        out["env_list_of_strings_submit"] = {
            "reward": state["reward"], "server_error": str(exc)[:200],
            "episode_left_open_until_closed": not state["submitted"]}
    ep = await sim.reset(gt_id)
    closed = await sim.close(ep["episode_id"])
    out["env_closed_without_submit"] = {"reward": closed["reward"], "submitted": closed["submitted"]}
    return out


async def _budget(sim: Simulator, inst: data.Instance) -> dict:
    ep = await sim.reset(inst.gt_id, budget=3)
    eid = ep["episode_id"]
    for _ in range(3):
        await sim.step(eid, "view_encounters", {"patient_id": inst.patient_id})
    over = await sim.step(eid, "view_encounters", {"patient_id": inst.patient_id})
    wrong = await sim.step(eid, "submit_summary", {"summary": "x"})
    final = await sim.step(eid, config.SUBMIT_TOOL, {"active_diagnoses": [], "chronic_conditions": []})
    return {"over_budget_observation": over["observation"], "over_budget_remaining": over["remaining"],
            "other_submit_tool_observation": wrong["observation"],
            "forced_submit": {"done": final["done"], "forced": final["info"].get("forced"),
                              "reward": final["reward"]}}


async def _crawl(sim: Simulator, inst: data.Instance) -> dict:
    """Read everything the tools will return for one patient; look for leaked labels."""
    ep = await sim.reset(inst.gt_id, budget=200)
    eid, pid = ep["episode_id"], inst.patient_id
    calls = [("open_chart", {"patient_id": pid}), ("view_encounters", {"patient_id": pid}),
             ("view_problem_list", {"patient_id": pid}), ("view_medications", {"patient_id": pid})]
    calls += [("view_results", {"patient_id": pid, "result_type": t}) for t in ("labs", "imaging", "pathology")]
    calls += [("view_encounter_detail", {"encounter_id": e["encounter_id"]}) for e in inst.encounters]
    calls += [("search_chart", {"patient_id": pid, "query": q})
              for q in ("assessment", "plan", "diagnosis", "impression")]
    calls += [("search_chart", {"patient_id": pid, "query": d["display_name"]}) for d in inst.gt_diagnoses[:3]]

    blob, outcome_sections, hidden_ids = [], 0, []
    for name, args in calls:
        out = await sim.step(eid, name, args)
        obs = out["observation"]
        blob.append(json.dumps(obs, default=str))
        outcome_sections += len(re.findall(r'"section_type": "(?:assessment|plan)"', blob[-1], flags=re.I))
    # Try to read assessment/plan sections directly by id (ids come from the raw database).
    conn = data.connect()
    marks = ",".join("?" * len(inst.encounters))
    hidden_ids = [r[0] for r in conn.execute(
        f"SELECT id FROM encounter_ehr_sections WHERE encounter_id IN ({marks}) "
        "AND lower(section_type) IN ('assessment', 'plan') LIMIT 4",
        [e["encounter_id"] for e in inst.encounters])]
    conn.close()
    direct_reads = []
    for sid in hidden_ids:
        out = await sim.step(eid, "view_section", {"section_id": sid})
        text = json.dumps(out["observation"], default=str)
        direct_reads.append("error" in out["observation"] if isinstance(out["observation"], dict) else False)
        blob.append(text)
    await sim.close(eid)

    text = "\n".join(blob)
    # A reference code counts as leaked only as a code token (e.g. "O90.3"), not as a substring of prose.
    codes = [d["icd10"] for d in inst.gt_diagnoses if d.get("icd10")]
    leaked = [c for c in codes if re.search(rf"(?<![A-Za-z0-9]){re.escape(c)}(?![A-Za-z0-9])", text)]
    return {"gt_id": inst.gt_id, "outcome_sections": outcome_sections, "codes_leaked": leaked,
            "hidden_section_direct_reads_blocked": direct_reads,
            "has_icd10_field": '"icd10' in text.lower()}


def name_exposure(instances: list[data.Instance]) -> dict:
    """How often a reference diagnosis's display name is readable in the visible chart.

    `carried_forward`: the name appears verbatim in a *later* encounter's section text
    (the "Active Problem List" that upstream documents as by-construction).
    `own_encounter_only`: it appears only in the encounter that established it.
    `last_encounter_dx`: reference diagnoses first established at the patient's final
    encounter, which have no later note to be carried into.
    """
    total = carried = own_only = anywhere = last_dx = last_dx_named = 0
    per_patient_carried = []
    for inst in instances:
        order = {e["encounter_id"]: e["encounter_order"] for e in inst.encounters}
        last_order = max(order.values()) if order else -1
        text_by_order = {e["encounter_order"]: "\n".join(s["section_text"] or "" for s in e["sections"]).lower()
                         for e in inst.encounters}
        n_carried = 0
        for d in inst.gt_diagnoses:
            total += 1
            name = d["display_name"].strip().lower()
            first = order.get(d.get("first_encounter_id"), -1)
            in_later = any(name in t for o, t in text_by_order.items() if o > first)
            in_own = name in text_by_order.get(first, "")
            carried += in_later
            n_carried += in_later
            own_only += in_own and not in_later
            anywhere += in_later or in_own
            if first == last_order:
                last_dx += 1
                last_dx_named += in_own
        per_patient_carried.append(n_carried / max(1, len(inst.gt_diagnoses)))
    return {
        "n_reference_diagnoses": total,
        "name_carried_forward_rate": carried / total,
        "name_in_own_encounter_only_rate": own_only / total,
        "name_readable_anywhere_rate": anywhere / total,
        "final_encounter_diagnoses_share": last_dx / total,
        "final_encounter_diagnoses_named_in_own_note_rate": last_dx_named / max(1, last_dx),
        "mean_per_patient_carried_forward": sum(per_patient_carried) / len(per_patient_carried),
    }


def terminology_rows() -> int | None:
    """ICD-10-CM rows loaded in the simulator's Postgres (the reward depends on them)."""
    try:
        out = subprocess.run(
            ["docker", "compose", "-f", str(config.ROOT / "docker" / "compose.yaml"), "exec", "-T", "postgres",
             "psql", "-U", "epic_sim", "-d", "epic_sim", "-At", "-c",
             "select count(*) from terminology_codes where system='icd10cm'"],
            capture_output=True, text=True, timeout=30, check=True)
        return int(out.stdout.strip())
    except Exception:  # noqa: BLE001 — not running under this compose file
        return None


async def run(crawl_n: int = 40, seed: int = 0) -> dict:
    instances = data.load_instances(("train", "public"))
    train = [i for i in instances if i.split == "train"]
    rng = random.Random(seed)
    sample = rng.sample(train, min(crawl_n, len(train)))
    async with Simulator() as sim:
        health = await sim.health()
        oracle = await _oracle_all(sim, instances)
        malformed = await _malformed(sim, train[0].gt_id)
        budget = await _budget(sim, train[0])
        sem = asyncio.Semaphore(8)

        async def crawl(inst):
            async with sem:
                return await _crawl(sim, inst)

        crawls = await asyncio.gather(*(crawl(i) for i in sample))

    leak = {
        "episodes_crawled": len(crawls),
        "episodes_with_outcome_sections": sum(c["outcome_sections"] > 0 for c in crawls),
        "episodes_with_reference_code_in_observations": sum(bool(c["codes_leaked"]) for c in crawls),
        "episodes_with_icd10_field": sum(c["has_icd10_field"] for c in crawls),
        "direct_hidden_section_reads": sum(len(c["hidden_section_direct_reads_blocked"]) for c in crawls),
        "direct_hidden_section_reads_blocked": sum(sum(c["hidden_section_direct_reads_blocked"]) for c in crawls),
        "examples": [c for c in crawls if c["codes_leaked"] or c["outcome_sections"]][:5],
    }
    terminology = terminology_rows()

    failures = []
    if oracle["n_below_1"]:
        failures.append(f"oracle scored below 1.0 on {oracle['n_below_1']} instances")
    for name, value in malformed.items():
        reward = value["reward"] if isinstance(value, dict) else value
        if isinstance(value, dict) and value.get("server_error"):
            failures.append(f"malformed case '{name}' crashed the scorer instead of scoring 0")
        elif reward != 0:
            failures.append(f"malformed case '{name}' did not score 0: {str(value)[:200]}")
    if not budget["forced_submit"]["done"] or "error" not in budget["over_budget_observation"]:
        failures.append("budget not enforced")
    if leak["episodes_with_outcome_sections"]:
        failures.append("assessment/plan sections visible in observations")
    if leak["direct_hidden_section_reads"] != leak["direct_hidden_section_reads_blocked"]:
        failures.append("assessment/plan sections readable by section_id")
    if terminology == 0:
        failures.append("terminology_codes has no ICD-10-CM rows; chart-neutral sets (and rewards) will differ")

    return {
        "health": health,
        "icd10cm_terminology_rows": terminology,
        "oracle": oracle,
        "malformed": malformed,
        "budget": budget,
        "label_exposure": leak,
        "name_exposure_train": name_exposure(train),
        "failures": failures,
    }
