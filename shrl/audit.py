"""Phase 2 task/reward audit: hand-written strategies scored with the official scorer.

Each strategy maps an instance to a submission. Strategies fall in three groups:

chart extraction (uses only what the policy can read)
    How much of the task is solved by copying what the chart already says.
    These need an ICD-10 code for each extracted name; `name_to_icd` is an exact
    lookup in the benchmark's own diagnosis table, so the scores are an upper
    bound on "copy the name and code it correctly".

priors (uses no chart at all)
    Frequent categories from the training portion, high-severity guesses.

oracle perturbations (uses the labels)
    What the reward is and is not sensitive to: padding, chart-neutral padding,
    acuity, code specificity, duplicates.

A high-scoring extraction strategy is a property of the task, not automatically
an exploit; see results/audit.md for the interpretation.
"""

from __future__ import annotations

import asyncio
import collections
import re
from typing import Callable

from shrl import breakdown, data, metrics
from shrl.client import Simulator

Strategy = Callable[[data.Instance], dict]

# The scorer's critical list (eval/scoring.py CRITICAL_CODES_3CHAR).
CRITICAL = ["A41", "I21", "I26", "I63", "I60", "I61", "J96", "N17", "K72", "E87", "T78"]
PROBLEM_LINE = re.compile(r"^\s*-\s*(.+?)\s*(\(diagnosed (\d{4}-\d{2}-\d{2})\))?\s*$")


def _submission(codes: list[tuple[str, str]], acuity: str = "acute") -> dict:
    """All entries in one list; the scorer ignores which list a prediction is in."""
    seen, entries = set(), []
    for code, name in codes:
        if code and code not in seen:
            seen.add(code)
            entries.append({"icd10": code, "name": name, "acuity": acuity})
    if acuity == "chronic":
        return {"active_diagnoses": [], "chronic_conditions": entries}
    return {"active_diagnoses": entries, "chronic_conditions": []}


def _oracle(inst: data.Instance) -> dict:
    out = {"active_diagnoses": [], "chronic_conditions": []}
    for d in inst.gt_diagnoses:
        out[d["list"]].append({"icd10": d["icd10"], "name": d["display_name"], "acuity": d["acuity"]})
    return out


def _problem_list_lines(inst: data.Instance) -> list[tuple[str, bool]]:
    """(name, has_diagnosed_date) for every 'Active Problem List' bullet in visible PMH sections."""
    seen: dict[str, bool] = {}
    for enc in inst.encounters:
        for sec in enc["sections"]:
            text = sec["section_text"] or ""
            if "Active Problem List" not in text:
                continue
            block = text.split("Active Problem List", 1)[1].split("\n\n", 1)[0]
            for line in block.splitlines():
                m = PROBLEM_LINE.match(line)
                if m:
                    seen[m.group(1)] = seen.get(m.group(1), False) or bool(m.group(2))
    return list(seen.items())


def _coded(names: list[str]) -> list[tuple[str, str]]:
    table = data.name_to_icd()
    return [(table[n.strip().lower()], n) for n in names if n.strip().lower() in table]


# -- chart extraction ---------------------------------------------------------

def profile_problem_list(inst: data.Instance) -> dict:
    """What view_problem_list returns (the profile's chronic conditions), coded by exact name."""
    names = [str(c if not isinstance(c, dict) else c.get("name") or "")
             for c in inst.profile.get("chronic_conditions") or []]
    return _submission(_coded(names), "chronic")


def note_problem_list_dated(inst: data.Instance) -> dict:
    """'Active Problem List' bullets carrying '(diagnosed <date>)': earlier encounters' diagnoses."""
    return _submission(_coded([n for n, dated in _problem_list_lines(inst) if dated]))


def note_problem_list_all(inst: data.Instance) -> dict:
    """Every 'Active Problem List' bullet in any note."""
    return _submission(_coded([n for n, _ in _problem_list_lines(inst)]))


def chart_mentions(inst: data.Instance) -> dict:
    """Every knowledge-graph diagnosis whose display name appears verbatim anywhere in the visible chart."""
    text = inst.visible_text().lower()
    hits = [(d["icd10_code"], d["display_name"]) for d in data.diagnosis_table()
            if len(d["display_name"]) >= 6 and d["display_name"].lower() in text]
    return _submission(hits)


# -- priors -------------------------------------------------------------------

def make_common(train: list[data.Instance], k: int) -> Strategy:
    counts = collections.Counter(data.category(d["icd10"]) for i in train for d in i.gt_diagnoses)
    top = [(c, c) for c, _ in counts.most_common(k)]
    return lambda inst: _submission(top)


def critical_guess(inst: data.Instance) -> dict:
    return _submission([(c, c) for c in CRITICAL])


# -- oracle perturbations -----------------------------------------------------

def make_padded(n: int) -> Strategy:
    """Oracle plus n wrong codes from an unused ICD-10 block (never chart-neutral)."""
    def strategy(inst: data.Instance) -> dict:
        out = _oracle(inst)
        out["active_diagnoses"] += [{"icd10": f"V9{i % 10}.{i // 10}", "name": "pad", "acuity": "acute"}
                                    for i in range(n)]
        return out
    return strategy


def make_neutral_padded(neutral: dict[int, list[str]]) -> Strategy:
    """Oracle plus one code in every chart-neutral category of the patient (expected to be free)."""
    def strategy(inst: data.Instance) -> dict:
        out = _oracle(inst)
        out["chronic_conditions"] += [{"icd10": c, "name": "neutral pad", "acuity": "chronic"}
                                      for c in neutral.get(inst.patient_id, [])]
        return out
    return strategy


def oracle_acuity_flipped(inst: data.Instance) -> dict:
    out = _oracle(inst)
    flipped = {"active_diagnoses": [], "chronic_conditions": []}
    for d in out["active_diagnoses"]:
        flipped["chronic_conditions"].append({**d, "acuity": "chronic"})
    for d in out["chronic_conditions"]:
        flipped["active_diagnoses"].append({**d, "acuity": "acute"})
    return flipped


def oracle_category_only(inst: data.Instance) -> dict:
    out = _oracle(inst)
    for key in out:
        out[key] = [{**d, "icd10": data.category(d["icd10"])} for d in out[key]]
    return out


def oracle_duplicated(inst: data.Instance) -> dict:
    out = _oracle(inst)
    return {k: v + v for k, v in out.items()}


def oracle_drop_last(inst: data.Instance) -> dict:
    """Oracle minus diagnoses first established at the final encounter (never carried forward)."""
    last = max(inst.encounters, key=lambda e: e["encounter_order"])["encounter_id"]
    out = {"active_diagnoses": [], "chronic_conditions": []}
    for d in inst.gt_diagnoses:
        if d.get("first_encounter_id") != last:
            out[d["list"]].append({"icd10": d["icd10"], "name": d["display_name"], "acuity": d["acuity"]})
    return out


def empty(inst: data.Instance) -> dict:
    return {"active_diagnoses": [], "chronic_conditions": []}


def malformed(inst: data.Instance) -> dict:
    return {"diagnosis": "I10"}


async def neutral_categories(sim: Simulator, instances: list[data.Instance]) -> dict[int, list[str]]:
    """Recover each patient's chart-neutral categories by probing the scorer.

    The sets are a deterministic function of the patient profile (which the
    policy can read), so this is a probe of the reward, not of the labels: one
    code per candidate category is submitted and `n_neutral_predictions` counts
    how many were waived.
    """
    candidates = sorted({data.category(d["icd10_code"]) for d in data.diagnosis_table()})
    sem = asyncio.Semaphore(8)

    async def one(inst: data.Instance) -> tuple[int, list[str]]:
        reference = {data.category(d["icd10"]) for d in inst.gt_diagnoses}
        found: list[str] = []

        async def probe(codes: list[str]) -> None:
            if not codes:
                return
            res = await sim.score(inst.gt_id, _submission([(c, c) for c in codes]))
            n = res["metrics"]["n_neutral_predictions"]
            if n == 0:
                return
            if n == len(codes):
                found.extend(codes)
                return
            mid = len(codes) // 2
            await probe(codes[:mid])
            await probe(codes[mid:])

        async with sem:
            await probe([c for c in candidates if c not in reference])
        return inst.patient_id, sorted(found)

    return dict(await asyncio.gather(*(one(i) for i in instances)))


async def run(eval_instances: list[data.Instance], train_instances: list[data.Instance],
              neutral_probe_n: int = 40) -> dict:
    async with Simulator() as sim:
        probe_set = eval_instances[:neutral_probe_n]
        neutral = await neutral_categories(sim, probe_set)

        strategies: list[tuple[str, str, Strategy, list[data.Instance]]] = [
            ("oracle", "oracle", _oracle, eval_instances),
            ("empty submission", "invalid", empty, eval_instances),
            ("malformed submission", "invalid", malformed, eval_instances),
            ("profile problem list (view_problem_list)", "extraction", profile_problem_list, eval_instances),
            ("note problem list, dated entries", "extraction", note_problem_list_dated, eval_instances),
            ("note problem list, all entries", "extraction", note_problem_list_all, eval_instances),
            ("all chart-mentioned diagnosis names", "extraction", chart_mentions, eval_instances),
            ("top-5 train categories", "prior", make_common(train_instances, 5), eval_instances),
            ("top-20 train categories", "prior", make_common(train_instances, 20), eval_instances),
            ("top-100 train categories", "prior", make_common(train_instances, 100), eval_instances),
            ("scorer's 11 critical categories", "prior", critical_guess, eval_instances),
            ("oracle + 5 wrong codes", "perturbation", make_padded(5), eval_instances),
            ("oracle + 20 wrong codes", "perturbation", make_padded(20), eval_instances),
            (f"oracle + all chart-neutral categories (n={len(probe_set)})", "perturbation",
             make_neutral_padded(neutral), probe_set),
            ("oracle, acuity flipped", "perturbation", oracle_acuity_flipped, eval_instances),
            ("oracle, 3-char categories only", "perturbation", oracle_category_only, eval_instances),
            ("oracle, every entry duplicated", "perturbation", oracle_duplicated, eval_instances),
            ("oracle minus final-encounter diagnoses", "perturbation", oracle_drop_last, eval_instances),
        ]

        rows = []
        for name, group, strategy, insts in strategies:
            preds = [strategy(i) for i in insts]
            results = await sim.score_batch([(i.gt_id, p) for i, p in zip(insts, preds)])
            records = [{"gt_id": i.gt_id, "n_pred": metrics.count_predictions(p),
                        "reward": r["reward"], "metrics": r["metrics"]}
                       for i, p, r in zip(insts, preds, results)]
            rows.append({"strategy": name, "group": group, **metrics.summarize(records, n_boot=300),
                         "visit_split": breakdown.split_recall(list(zip(insts, preds)))})

    sizes = [len(v) for v in neutral.values()]
    return {
        "n_eval": len(eval_instances),
        "rows": rows,
        "chart_neutral": {
            "patients_probed": len(sizes),
            "mean_categories": sum(sizes) / len(sizes) if sizes else 0,
            "max_categories": max(sizes) if sizes else 0,
            "patients_with_none": sum(s == 0 for s in sizes),
        },
    }
