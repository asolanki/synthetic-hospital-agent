"""Stage 2 inputs: candidate (name, code) pairs with up to two patient contexts each. Train + public only."""
import json, sqlite3, sys, re, random, collections
ROWS, DB, OUT = sys.argv[1:4]
rows = json.load(open(ROWS))
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
inst = collections.defaultdict(list)
for gid, pid, split, g in c.execute("select gt_id, patient_id, split, ground_truth from benchmark_ground_truth where task='patient_diagnosis' and split in ('train','public') order by gt_id"):
    g = json.loads(g)
    for d in g.get("active_diagnoses", []) + g.get("chronic_conditions", []):
        if d.get("excluded_nondiagnostic"): continue
        inst[((d.get("display_name") or "").strip(), (d.get("icd10") or "").strip())].append((pid, split, d.get("first_encounter_id"), d.get("acuity")))
def sec(eid, t, n):
    r = c.execute("select section_text from encounter_ehr_sections where encounter_id=? and section_type=?", (eid, t)).fetchone()
    return " ".join((r[0] if r else "").split())[:n]
def ctx(pid, split, eid, acuity):
    age, sex = c.execute("select age, sex from longitudinal_patients where patient_id=?", (pid,)).fetchone()
    cc, edate, etype = c.execute("select chief_complaint, encounter_date, encounter_type from longitudinal_encounters where encounter_id=?", (eid,)).fetchone()
    return {"age_at_first_visit": age, "sex": sex, "encounter_type": etype, "chief_complaint": (cc or "")[:250],
            "hpi": sec(eid, "hpi", 700), "assessment": sec(eid, "assessment", 350) or sec(eid, "plan", 350), "reference_acuity": acuity}
rng = random.Random(0)
flag = [r for r in rows if r["cls"][:2] in ("A:", "B:")]
unclear = [r for r in rows if r["cls"].startswith("unclear")]
okpool = [r for r in rows if r["cls"].startswith("ok")]
control = rng.sample(okpool, 100)
cands = []
for stratum, group in (("flagged", flag), ("unclear", unclear), ("control", control)):
    for r in group:
        insts = inst[(r["name"], r["ref_code"])]
        cands.append({"stratum": stratum, "name": r["name"], "ref_code": r["ref_code"], "ref_title": r["ref_title"],
                      "n_references": r["n"], "n_public": r["public"],
                      "book_top8": [f"{c_} {t}" for c_, t in r["book_top"]],
                      "patients": [ctx(*x) for x in insts[:2]]})
rng.shuffle(cands)
for i, x in enumerate(cands): x["id"] = f"P{i:04d}"
json.dump(cands, open(OUT, "w"), indent=1)
print(len(cands), collections.Counter(x["stratum"] for x in cands))
print(json.dumps(cands[0], indent=1)[:1800])
