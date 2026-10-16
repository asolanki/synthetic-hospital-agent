# Label audit: protocol (written before any verdict)

Written 2026-10-08 22:00 EDT by the cloud Claude session, before any judge call. Exploratory; not part of the training/eval protocol. Heldout is never read.

**Question.** How often is a patient-diagnosis reference code wrong at the level the scorer checks (3-character category; 5 characters for S/T codes), and how much would correcting those labels change each arm's public score?

**Scope.** Every distinct (reference name, reference code) pair in the `train` (our train + dev) and `public` splits of `benchmark_v1.3.db`: 1,972 pairs, 3,571 reference instances. `heldout` is excluded.

**Stage 1, deterministic scan** (`audit/label_scan.py`). Each pair is checked against the frozen code book (`search-index-2025-v3`, CMS ICD-10-CM 2025):
- A: the name is an official title of a code in another category (24 pairs).
- B: the reference code's title shares almost nothing with the name, and the book's first result fits the name in another category (34 pairs).
- unclear: the reference category is not in the book's top 8 (353 pairs).
- ok / ok-ish: the book's top 1 / top 8 contains the reference category (1,561 pairs).

**Stage 2, judged.** Candidates: all A, B and unclear pairs, plus a control of 100 ok/ok-ish pairs (random, seed 0). Each judge sees, for up to two patient instances of a pair: age, sex, chief complaint, history of present illness and assessment of the encounter where the diagnosis was first made (truncated), the reference code with its official title, and the book's top 8 results for the name. Judges never see any model's output.
- Verdict per pair, at the scoring category: `correct`, `defensible` (a qualified coder could choose the reference category, e.g. combination codes, pregnancy or newborn precedence, acute vs chronic variants), `wrong` (the category describes a different condition or contradicts the name and chart; give the right code), or `unsure`.
- Two independent Sonnet judges with different batch orders. A pair is `wrong` only if both say so, or if a third, stronger judge (Opus) says so on a disagreement.

**Outputs.** `results/raw/audit/label_audit_{candidates,verdicts,summary}.json`.
- Error rate per stratum, and an estimate for all references (the control sample stands in for the ok strata).
- A list of wrong labels with corrected codes, for the benchmark authors.
- Sensitivity check: every public arm rescored with corrected labels. This is exploratory and replaces no reported number.

**Predictions.**
- Wrong (category level): A+B about 50%; unclear about 10–15%; control under 2%.
- Overall: 1–5% of train+public references (point guess 2.5%).
- Rescoring: tool-SFT changes by −0.01 to 0. Luna, Kimi and Luna + book gain 0 to +0.02. The tool-SFT − Luna gap shrinks by less than 0.02. That would mean most of the house-style gap is precision (what to list), not label errors.
