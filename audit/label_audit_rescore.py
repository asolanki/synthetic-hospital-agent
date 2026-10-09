"""Rescore saved public submissions with the paper's own scorer (eval.scoring) and chart-neutral sets
(scripts/chart_neutral_sets.py), optionally with corrected reference codes. Public split only."""
import json, sqlite3, sys, re, importlib.util, math, xml.etree.ElementTree as ET
UP, DB, TAB, RES = sys.argv[1:5]
sys.path.insert(0, UP)
from eval.scoring import compute_all_metrics
spec = importlib.util.spec_from_file_location("cns", f"{UP}/scripts/chart_neutral_sets.py"); cns = importlib.util.module_from_spec(spec); sys.modules["cns"] = cns; spec.loader.exec_module(cns)
mem = sqlite3.connect(":memory:", uri=True); mem.execute(f"attach database 'file:{DB}?mode=ro' as b")
mem.execute("create table diagnoses as select * from b.diagnoses"); mem.execute("create table longitudinal_patients as select * from b.longitudinal_patients")
mem.execute("create table terminology_codes(system text, code text, display text)")
cats = [(d.findtext("name"), d.findtext("desc")) for d in ET.parse(TAB).getroot().iter("diag") if d.findtext("name") and len(d.findtext("name")) == 3]
mem.executemany("insert into terminology_codes values ('icd10cm', ?, ?)", cats)
class Cur:  # psycopg-style cursor over sqlite (the script only uses execute/fetchall)
    def __init__(s): s.c = mem.cursor()
    def execute(s, q, p=()): s.c.execute(q.replace("%s", "?"), p)
    def fetchall(s): return s.c.fetchall()
    def fetchone(s): return s.c.fetchone()
neutral, _, _ = cns.neutral_sets(Cur())
src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
GT = {gid: (pid, json.loads(g)) for gid, pid, g in src.execute("select gt_id, patient_id, ground_truth from benchmark_ground_truth where task='patient_diagnosis' and split='public'")}
def reward(pred, gt, pid):
    g = dict(gt); g["_neutral_categories"] = sorted(neutral.get(pid, set()))
    m = compute_all_metrics("patient_diagnosis", [pred if isinstance(pred, dict) else {}], [g])
    r = m["weighted_problem_list_f1_neutral"]
    return 0.0 if r is None or (isinstance(r, float) and math.isnan(r)) else min(1.0, max(0.0, float(r)))
def corrected(gt, fixes):
    g = json.loads(json.dumps(gt))
    for k in ("active_diagnoses", "chronic_conditions"):
        for d in g.get(k, []):
            new = fixes.get(((d.get("display_name") or "").strip(), (d.get("icd10") or "").strip()))
            if new: d["icd10"] = new
    return g
def score_arm(name, fixes=None):
    recs = json.load(open(f"{RES}/{name}.json"))["records"]
    out = {}
    for r in recs:
        pid, gt = GT[r["gt_id"]]
        out[r["gt_id"]] = reward(r.get("submission") or {}, corrected(gt, fixes) if fixes else gt, pid)
    return recs, out
if __name__ == "__main__":
    for name in sys.argv[5:]:
        recs, out = score_arm(name)
        diffs = [abs(out[r["gt_id"]] - r["reward"]) for r in recs]
        print(f"{name}: recorded {sum(r['reward'] for r in recs)/len(recs):.4f} rescored {sum(out.values())/len(out):.4f} max |diff| {max(diffs):.2e} mismatches>1e-9: {sum(d>1e-9 for d in diffs)}")
