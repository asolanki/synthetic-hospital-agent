"""Deterministic first pass: reference (name, code) pairs whose code looks inconsistent with the name,
judged only against the public CMS ICD-10-CM 2025 code book. Train (incl. our dev) + public only; never heldout."""
import json, sqlite3, sys, re, collections, importlib.util
REPO, DB = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("icd10", f"{REPO}/training/environment/synthetic_hospital_dx/icd10.py")
icd10 = importlib.util.module_from_spec(spec); sys.modules["icd10"] = icd10; spec.loader.exec_module(icd10)
idx = json.load(open(f"{REPO}/runs/icd10cm/search-index-2025-v3.json"))
book = icd10.CodeBook(idx["titles"], [tuple(d) for d in idx["documents"]])
titles = idx["titles"]
key = lambda c: (lambda n: n[:5] if n[:1] in "ST" else n[:3])(re.sub(r"[^A-Z0-9]", "", (c or "").upper()))
def title(code):
    c = (code or "").upper().strip()
    while c and c not in titles: c = c[:-1].rstrip(".")
    return titles.get(c, "")
ntok = lambda s: set(icd10.tokens(s or "")) - {"unspecified", "other", "specified", "type", "initial", "encounter", "subsequent", "sequela"}
jac = lambda a, b: len(a & b) / len(a | b) if a | b else 0.0
norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))
title_to_codes = collections.defaultdict(set)
for c, t in titles.items(): title_to_codes[norm(t)].add(c)
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
pairs = collections.defaultdict(lambda: {"n": 0, "splits": collections.Counter()})
for split, g in c.execute("select split, ground_truth from benchmark_ground_truth where task='patient_diagnosis' and split in ('train','public')"):
    g = json.loads(g)
    for d in g.get("active_diagnoses", []) + g.get("chronic_conditions", []):
        if d.get("excluded_nondiagnostic"): continue
        p = pairs[(d["display_name"].strip(), d["icd10"].strip())]; p["n"] += 1; p["splits"][split] += 1
out = collections.Counter(); rows = []
for (name, code), p in pairs.items():
    rk = key(code); rt = title(code)
    exact = title_to_codes.get(norm(name), set())
    hits = book.search(name, 8) if hasattr(book, "search") else book.query(name, 8)
    top = hits[0] if hits else None
    top_keys = [key(h.code) for h in hits]
    j_ref = jac(ntok(name), ntok(rt))
    j_top = jac(ntok(name), ntok(top.title)) if top else 0
    if exact and rk not in {key(x) for x in exact}:
        cls = "A: name is an official title of a code in another category"
    elif j_ref < 0.15 and top and key(top.code) != rk and j_top >= 0.5:
        cls = "B: reference code's title unrelated to name; book's top hit fits"
    elif rk in top_keys[:1]: cls = "ok: book top-1 agrees"
    elif rk in top_keys: cls = "ok-ish: reference category in book top-8"
    else: cls = "unclear: reference category not in book top-8"
    out[cls] += p["n"]; out["_pairs:" + cls[:2]] += 1
    rows.append(dict(cls=cls, name=name, ref_code=code, ref_title=rt, n=p["n"], public=p["splits"]["public"],
                     book_top=[(h.code, h.title) for h in hits[:8]]))
tot = sum(p["n"] for p in pairs.values())
print(f"distinct (name, code) pairs: {len(pairs)}; reference instances: {tot}")
for k, v in sorted(out.items()):
    if not k.startswith("_"): print(f"  {k:70s} {v:5d} ({v/tot:.1%})   distinct pairs: {out['_pairs:'+k[:2]]}")
json.dump(rows, open("/tmp/claude-0/-home-claude/6aaf2c65-2252-5bc6-b7c3-c92efcc691e4/scratchpad/label_scan_rows.json", "w"), indent=1)
for cls in ("A:", "B:"):
    sel = sorted([r for r in rows if r["cls"].startswith(cls)], key=lambda r: -r["n"])
    print(f"\n{cls} {len(sel)} pairs; top by frequency:")
    for r in sel[:12]:
        print(f"  [{r['n']}x] {r['name']}  ->  ref {r['ref_code']} '{r['ref_title']}'  | book: {r['book_top'][0][0]} '{r['book_top'][0][1]}'")
