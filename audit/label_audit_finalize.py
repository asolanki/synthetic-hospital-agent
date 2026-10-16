"""Combine the three judges' verdicts, estimate the error rate, and rescore every run with corrected labels.

Run from audit/ (it imports label_audit_rescore as `rescore`), next to candidates.json and verdicts/*.jsonl:
    python label_audit_finalize.py UPSTREAM_REPO BENCHMARK_DB ICD10CM_TABULAR_XML RESULTS_DIR

Example, from the repo root:
    cd audit && python label_audit_finalize.py ../external/synthetic_hospital \
        ../external/synthetic_hospital/benchmark_v1.3.db ../runs/icd10cm/icd10cm_tabular_2025.xml ../results/raw/training
"""
import json, glob, collections, re, random, statistics, sys, importlib.util
spec = importlib.util.spec_from_file_location("rescore", __file__.replace("label_audit_finalize.py", "label_audit_rescore.py"))
R = importlib.util.module_from_spec(spec); sys.modules["rescore"] = R; spec.loader.exec_module(R)
SUMMARY = {"rates": {}, "rescore": {}, "contrasts": {}, "reproduce_key_mistakes": {}}
key = lambda c: (lambda n: n[:5] if n[:1] in "ST" else n[:3])(re.sub(r"[^A-Z0-9]", "", (c or "").upper()))
cand = {x["id"]: x for x in json.load(open("candidates.json"))}
V = collections.defaultdict(dict)
for f in glob.glob("verdicts/*.jsonl"):
    for l in open(f):
        if l.strip(): r = json.loads(l); V[f.split("/")[-1][:2]][r["id"]] = r
def eff(r, ref):
    if r is None: return None
    return "same-cat" if r["verdict"] == "wrong" and key(r.get("correct_code")) == key(ref) else r["verdict"]
final = {}
for i, x in cand.items():
    a, b, c = (eff(V[j].get(i), x["ref_code"]) for j in ("j1", "j2", "j3"))
    primary = (a == "wrong" and b == "wrong") or (("wrong" in (a, b)) and not (a == b == "wrong") and c == "wrong")
    strict = a == b == c == "wrong"
    if c == "wrong": fix = V["j3"][i]["correct_code"]
    elif a == "wrong": fix = V["j1"][i]["correct_code"]
    elif b == "wrong": fix = V["j2"][i]["correct_code"]
    else: fix = None
    final[i] = dict(primary=primary, strict=strict, fix=fix, j=(a, b, c))
# ---- rates
refs_by_stratum = {"flagged": 85, "unclear": 539, "ok": 2947}
print("WRONG LABELS (primary rule: both Sonnet judges, or Opus on a split; strict: all three)")
for s in ("flagged", "unclear", "control"):
    ids = [i for i in cand if cand[i]["stratum"] == s]
    p = [i for i in ids if final[i]["primary"]]; st = [i for i in ids if final[i]["strict"]]
    rp = sum(cand[i]["n_references"] for i in p); rt = sum(cand[i]["n_references"] for i in ids)
    print(f"  {s:8s} pairs {len(p)}/{len(ids)} ({len(p)/len(ids):.0%}); strict {len(st)}; references {rp}/{rt} ({rp/rt:.1%})")
    SUMMARY["rates"][s] = {"pairs_wrong": len(p), "pairs": len(ids), "pairs_wrong_strict": len(st), "references_wrong": rp, "references": rt}
ctrl = [i for i in cand if cand[i]["stratum"] == "control"]
def ok_rate(sample): 
    n = sum(cand[i]["n_references"] for i in sample); return sum(cand[i]["n_references"] for i in sample if final[i]["primary"]) / n
judged_wrong_refs = sum(cand[i]["n_references"] for i in cand if cand[i]["stratum"] != "control" and final[i]["primary"])
rng = random.Random(0); boots = sorted(ok_rate([rng.choice(ctrl) for _ in ctrl]) for _ in range(1000))
tot = 3571
for lab, r in (("point", ok_rate(ctrl)), ("lo", boots[24]), ("hi", boots[974])):
    print(f"  estimated wrong share of ALL train+public references ({lab}): {(judged_wrong_refs + r*refs_by_stratum['ok'])/tot:.1%}  (ok-stratum rate {r:.1%})")
    SUMMARY["rates"]["overall_estimate_" + lab] = (judged_wrong_refs + r*refs_by_stratum['ok'])/tot
wrong_ids = [i for i in cand if final[i]["primary"]]
pub = sum(cand[i]["n_public"] for i in wrong_ids); print(f"  public references with a primary-wrong label (judged strata + control sample): {pub} of 705")
# ---- rescoring
fixes = {(cand[i]["name"], cand[i]["ref_code"]): final[i]["fix"] for i in wrong_ids if cand[i]["stratum"] != "control" or True}
arms = {"untrained 4B": "qwen3-4b-paper-public", "untrained 4B + book": "qwen3-4b-icd10-public", "SFT 3 ep": "qwen3-4b-sft-paper-public",
        "SFT 3 ep + RL": "qwen3-4b-rl-paper-public", "continued SFT": "qwen3-4b-sft-continued-paper-public", "SFT 8 ep": "qwen3-4b-sft-sweep-e8-paper-public",
        "tool-SFT": "qwen3-4b-sft-icd10-public", "Qwen3-30B": "qwen3-30b-env-public-t0", "Qwen3-30B + book": "qwen3-30b-icd10-public-t0",
        "Kimi K2.5": "kimi-k2.5-env-public", "GPT-6 Luna": "gpt-6-luna-env-public", "Luna + book": "gpt-6-luna-icd10-env-public", "Gemma 3 27B": "gemma-3-27b-env-public"}
S = {}
print("\nRESCORE (public mean reward): original -> corrected labels (delta)")
for a, f in arms.items():
    recs, orig = R.score_arm(f); _, corr = R.score_arm(f, fixes); S[a] = (orig, corr)
    SUMMARY["rescore"][a] = {"original": statistics.mean(orig.values()), "corrected": statistics.mean(corr.values())}
    print(f"  {a:20s} {statistics.mean(orig.values()):.3f} -> {statistics.mean(corr.values()):.3f} ({statistics.mean(corr.values())-statistics.mean(orig.values()):+.3f})")
ids = sorted(S["tool-SFT"][0])
def ci(f):
    rng = random.Random(0); v = sorted(f([rng.choice(ids) for _ in ids]) for _ in range(1000)); return f(ids), v[24], v[974]
print("\nKEY CONTRASTS, original vs corrected labels")
for a, b in (("tool-SFT", "GPT-6 Luna"), ("tool-SFT", "Luna + book"), ("tool-SFT", "Kimi K2.5"), ("tool-SFT", "SFT 8 ep"), ("SFT 3 ep + RL", "SFT 3 ep")):
    for lab, k in (("orig", 0), ("corr", 1)):
        d = ci(lambda s: statistics.mean(S[a][k][i] - S[b][k][i] for i in s))
        print(f"  {a} - {b} [{lab}]: {d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})")
        SUMMARY["contrasts"][f"{a} - {b} [{lab}]"] = d
# ---- who reproduces the key's mistakes? on public references with a wrong label
GT = R.GT; wrongset = {(cand[i]["name"], cand[i]["ref_code"]): final[i]["fix"] for i in wrong_ids}
print("\nON PUBLIC REFERENCES WITH A WRONG LABEL: share of episodes submitting the reference's (wrong) category vs the corrected category")
for a in ("SFT 8 ep", "tool-SFT", "Kimi K2.5", "GPT-6 Luna", "Luna + book"):
    recs = {r["gt_id"]: r for r in json.load(open(f"/mnt/user-data/uploads/synthetic-hospital-rl/results/raw/training/{arms[a]}.json"))["records"]}
    n = w = c = 0
    for gid, (pid, gt) in GT.items():
        sub = recs[gid].get("submission") or {}; keys = {key(p.get("icd10")) for p in (sub.get("active_diagnoses") or []) + (sub.get("chronic_conditions") or [])}
        for d in gt.get("active_diagnoses", []) + gt.get("chronic_conditions", []):
            k = ((d.get("display_name") or "").strip(), (d.get("icd10") or "").strip())
            if k in wrongset: n += 1; w += key(k[1]) in keys; c += key(wrongset[k]) in keys
    SUMMARY["reproduce_key_mistakes"][a] = {"n": n, "submitted_wrong_reference_category": w, "submitted_corrected_category": c}
    print(f"  {a:12s} n={n}: wrong reference category {w} ({w/n:.0%}), corrected category {c} ({c/n:.0%})")
json.dump({"final": final, "fixes": [[k[0], k[1], v] for k, v in fixes.items()]}, open("final.json", "w"), indent=1)

SUMMARY["method"] = ("Primary: wrong if both Sonnet judges say wrong (cross-category), or one does and Opus agrees on adjudication. Strict: all three. "
  "Overall estimate: judged strata counted exactly; ok strata (1,561 pairs, 2,947 references) estimated from a 100-pair control sample, reference-weighted, 1,000 bootstrap resamples. "
  "Rescoring replaces only identified wrong labels (judged strata + control sample), so it undercorrects by the unjudged ok-stratum errors. Scorer: upstream eval.scoring.compute_all_metrics with scripts/chart_neutral_sets.py; reproduces every recorded public reward exactly before correction.")
json.dump(SUMMARY, open("summary.json", "w"), indent=1)
