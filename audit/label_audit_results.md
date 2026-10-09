# Label audit: results

Run 2026-10-08, 22:00–23:15 EDT, by the cloud Claude session, following `audit/label_audit_protocol.md` (written before any verdict). Exploratory: it changes no reported number. Train (our train + dev) and public only; heldout was never read.

## Answer

1. **About 6% of patient-diagnosis reference codes are wrong at the level the scorer checks:** 5.7% of train + public references, 95% CI 4.1–7.9%. The audit names 101 wrong (name, code) pairs covering 136 references; all three judges agree on 80 of them. Examples:
   - "Hepatic adenoma" is K76.0 (fatty liver).
   - "Anorectal fistula" is N36.0 (urethral fistula).
   - "Korsakoff syndrome" is E51.1 (beriberi).
   - Serotonin syndrome appears under four different wrong categories.
   - One reference "diagnosis" is "African American Ethnicity", coded Z59.9 (problem related to housing and economic circumstances).
2. **10 references (0.3%) use WHO ICD-10 codes that don't exist in ICD-10-CM.** These are J46.9 (status asthmaticus), B23.0 (acute HIV infection), I64 (stroke), G41.0, U04.9 (SARS), I98.21 and Y48.4. A model that codes correctly in ICD-10-CM can never match them. This is a deterministic count, with no judge involved.
3. **Correcting the wrong labels changes no conclusion.** Every public arm moves by at most 0.01, and every paired contrast keeps its verdict (table below).
4. **Models with the code book code the mislabeled diagnoses correctly more often than they copy the error.** On the 27 public references with an identified wrong label:
   - tool-SFT submitted the corrected category 56% of the time and the wrong reference category 26%;
   - Luna + book: 52% vs 11%;
   - 8-epoch SFT without the book: 22% vs 30%.

   So the 4B's lead over Luna is not learned label errors; it is mostly which diagnoses to list (`milestone5`, house style). The book makes models more correct than the answer key, and the scorer penalizes that.

## Error rates

| Stratum (stage 1 scan) | Pairs judged | Wrong pairs (strict) | References | Wrong references |
|---|--:|--:|--:|--:|
| Flagged: name fits another category | 58 | 24 (16) | 85 | 39 (46%) |
| Unclear: reference category not in the book's top 8 | 353 | 73 (60) | 539 | 93 (17%) |
| Control: random 100 of the 1,561 pairs the book agrees with | 100 | 4 (4) | 166 | 4 (2.4%) |
| **All train + public references (estimate)** | | | 3,571 | **5.7%** (4.1–7.9%) |

"Wrong" means both Sonnet judges said wrong at the scoring category (3 characters; 5 for S and T codes), or one did and Opus agreed on adjudication. "Strict" means all three said wrong. Opus re-judged all 88 double-Sonnet "wrong" pairs blind as an extra check: it agreed on 80, called 6 defensible and was unsure on 2. On the 30 split pairs it said wrong 13 times, defensible 12 and unsure 5. Wrong-vs-not agreement between the two Sonnet passes was 94.1%. The overall estimate counts the judged strata exactly and extrapolates the control rate (reference-weighted, 1,000 bootstrap resamples) to the 2,947 references in the agreeing strata.

## Rescoring with corrected labels (public, mean reward)

| Arm | Original | Corrected | Change |
|---|--:|--:|--:|
| Untrained 4B | 0.158 | 0.157 | −0.000 |
| Untrained 4B + book | 0.420 | 0.419 | −0.001 |
| SFT 3 epochs | 0.379 | 0.382 | +0.003 |
| SFT 3 epochs + RL | 0.387 | 0.397 | +0.010 |
| Continued SFT | 0.425 | 0.427 | +0.001 |
| SFT 8 epochs | 0.520 | 0.518 | −0.002 |
| **Tool-SFT** | 0.726 | 0.732 | +0.005 |
| Qwen3-30B | 0.427 | 0.428 | +0.001 |
| Qwen3-30B + book | 0.614 | 0.622 | +0.008 |
| Kimi K2.5 | 0.693 | 0.687 | −0.006 |
| GPT-6 Luna | 0.654 | 0.660 | +0.005 |
| Luna + book | 0.646 | 0.652 | +0.006 |
| Gemma 3 27B | 0.235 | 0.227 | −0.008 |

| Paired contrast (public) | Original labels | Corrected labels |
|---|---|---|
| Tool-SFT − GPT-6 Luna | +0.072 (+0.033 to +0.111) | +0.072 (+0.033 to +0.113) |
| Tool-SFT − Luna + book | +0.081 (+0.040 to +0.120) | +0.080 (+0.038 to +0.121) |
| Tool-SFT − Kimi K2.5 | +0.034 (−0.014 to +0.079) | +0.045 (−0.002 to +0.093) |
| Tool-SFT − SFT 8 epochs | +0.206 (+0.160 to +0.251) | +0.214 (+0.169 to +0.258) |
| SFT 3 epochs + RL − SFT 3 epochs | +0.008 (−0.014 to +0.032) | +0.015 (−0.010 to +0.042) |

The rescorer uses upstream's own `eval.scoring.compute_all_metrics` and `scripts/chart_neutral_sets.py`. Before any correction it reproduces every recorded per-patient reward exactly (0 mismatches on the tool-SFT and Luna public runs). Only the identified wrong labels are replaced, so the correction is partial: it misses the estimated ~2.4% error rate among the unjudged agreeing pairs.

## Predictions (from the protocol)

| Prediction | Result |
|---|---|
| Flagged pairs about 50% wrong | 41% of pairs, 46% of references: close |
| Unclear 10–15% | 21% of pairs, 17% of references: **higher** |
| Control under 2% | 4% of pairs, 2.4% of references: **slightly higher** |
| Overall 1–5% (point 2.5%) | 5.7% (4.1–7.9%): **higher than predicted** |
| Tool-SFT changes by −0.01 to 0 | +0.005: **wrong direction**. It codes mislabeled diagnoses correctly more often than it copies the error. |
| Frontier models gain 0 to +0.02 | Luna +0.005 and Luna + book +0.006 as predicted; Kimi −0.006, **wrong** |
| Tool-SFT − Luna gap shrinks by less than 0.02 | Unchanged: right |

## Caveats

- The judges are LLMs, not certified coders. Some verdicts are borderline: for example, "venous air embolism" coded T79.0 (traumatic air embolism) in a patient after a car crash, and intent characters on poisoning codes (T61.04 vs T61.01), which the scorer's 5-character rule for T codes counts as different. The user spot-check (`audit/label_audit_spotcheck.md`) is pending.
- The judges see only excerpts of each chart (chief complaint, history of present illness, a short assessment) and up to two patients per pair.
- The control sample is small (100 pairs), so the overall estimate's interval is wide.

## Files

`audit/label_scan.py` (stage 1), `audit/label_audit_build_candidates.py`, `audit/label_audit_judge_instructions.md`, `audit/label_audit_rescore.py`, `audit/label_audit_finalize.py`; `results/raw/audit/label_audit_{candidates,verdicts,summary}.json` and `label_audit_wrong_labels.csv`, the list for the benchmark authors.

The scripts ran in the cloud workspace and are saved as run, with that workspace's paths (upstream checkout, `benchmark_v1.3.db`, the CMS tabular XML) and module names (`label_audit_finalize.py` imports the rescorer as `rescore`). To rerun locally, point them at `external/synthetic_hospital` and `runs/icd10cm/`.

## Addendum, 2026-10-09: the told-vs-taught arms

Run locally by Claude Code after the audit, with this rescorer and the 101 corrections in `label_audit_wrong_labels.csv`. It reproduces every recorded public reward exactly (0 mismatches) on six arms: tool-SFT, Luna, Luna + book, Kimi, Luna + book + style, and the told 8-epoch SFT. The rescorer needed one portability fix, `uri=True` on its in-memory connection; the cloud environment's SQLite did not require it. The contrasts above reproduce to within 0.001 at the CI ends.

| Public | Original | Corrected | Change |
|---|--:|--:|--:|
| Luna + book + house style | 0.777 | 0.795 | **+0.018** |
| 8-epoch SFT + book, told | 0.514 | 0.514 | −0.000 |

| Paired contrast (public) | Original labels | Corrected labels |
|---|---|---|
| Tool-SFT − Luna + book + style | −0.051 (−0.090 to −0.012) | **−0.064** (−0.102 to −0.023) |
| Luna + book + style − Luna + book | +0.131 (+0.098 to +0.164) | +0.143 (+0.110 to +0.177) |

On the 27 public references with a wrong label, Luna + book + style submitted the corrected category **81%** of the time (22) and the key's wrong category 11% (3). The told 8-epoch SFT, which never looks codes up, submitted them 26% (7) and 30% (8) of the time. Correcting the labels helps the house-style arm more than any other, and widens its lead over the 4B. Output: `results/raw/audit/label_audit_rescore_told_arms.json`.
