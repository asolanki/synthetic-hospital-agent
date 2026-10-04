# Audit of the other official tasks

Date: 2026-10-02. Question: is there an official task in Synthetic Hospital v1.3 where *using the
chart* is the bottleneck, so that RL on tool use has something to add? All numbers are on the dev
patients (120 patients of the fixed split) with the official scorer through `/score`.
Reproduce with `uv run python -m audit.other_tasks <task>`; raw outputs are in `results/raw/other_tasks/`.

## Answer

No. None of the three is a better vehicle than `patient_diagnosis`.

| Task | Reward | What the audit found | Verdict |
|---|---|---|---|
| `evidence_retrieval` | precision@5 over chart sections | Relevance labels are a rule on section type. Sorting sections by type scores 0.98 without reading anything. | Solved by a one-line rule |
| `context_summarization` (whole patient) | recall of must-include findings | Recall only, no length or precision term. Pasting the entire chart scores 0.70 and longer is always better. | Reward is exploitable; RL would learn to paste |
| `imaging_indication` | concept F1 against an LLM-written reference question | No shortcut. But the chart is worth about 0.08–0.10, prior visits add nothing, and a 235B model scores about the same as a 30B one. | Honest task, little headroom, headroom is wording |

## evidence_retrieval (n=120)

| Strategy | Precision@5 (reward) | nDCG@10 |
|---|--:|--:|
| Empty | 0.000 | 0.000 |
| 5 random sections | 0.708 | 0.349 |
| First 5 sections of the chart | 0.703 | 0.376 |
| All sections, sorted by section type (HPI first) | 0.980 | 0.822 |

Every relevance judgment in the released database has `source = rule_based` and a rationale of the
form `section_type=...`. 71% of all sections are graded relevant, and HPI sections are relevant
99.7% of the time. The task prompt also hands the model the patient's diagnoses as the query.

## context_summarization, whole-patient variant (n=120)

| Strategy | Reward | Mean words |
|---|--:|--:|
| Empty | 0.000 | 0 |
| Chief complaints only | 0.178 | 52 |
| Every HPI, concatenated | 0.236 | 224 |
| HPI + exam + labs + imaging, concatenated | 0.601 | 493 |
| The entire chart, pasted | 0.702 | 1,047 |

The reward metric is named `clinical_f1` but is recall of must-include findings in the summary text.
Nothing penalizes length. The specialty-conditioned variant was not audited.

## imaging_indication (n=161 orders over 66 patients)

Hand-written strategies:

| Strategy | Concept F1 (reward) | Precision | Recall |
|---|--:|--:|--:|
| Oracle (reference question) | 1.000 | 1.000 | 1.000 |
| Empty | 0.000 | | |
| The order's indication, verbatim | 0.184 | 0.529 | 0.116 |
| Chief complaint of the encounter | 0.174 | 0.305 | 0.135 |
| Chief complaint + indication | 0.208 | 0.304 | 0.177 |
| HPI of the encounter, verbatim | 0.152 | 0.140 | 0.198 |
| Another patient's reference question | 0.061 | 0.070 | 0.060 |
| [uses labels] the encounter's reference diagnosis name | 0.208 | 0.560 | 0.135 |
| [uses labels] the reference differential's diagnosis names | 0.370 | 0.411 | 0.368 |
| [uses labels] reference question + 5 unrelated diagnoses | 0.634 | 0.473 | 0.999 |

Models, by how much of the chart they are given:

| Condition | Qwen3-30B | Qwen3-235B |
|---|--:|--:|
| Single turn, order only (no chart) | 0.327 | 0.346 |
| Single turn, order + current encounter note | 0.410 | 0.453 |
| Single turn, order + full chart up to the order | 0.409 | 0.434 |
| Tool-using agent in `/env` (paper prompt) | 0.374 | not run |

Reading:

- No extraction or padding shortcut exists; padding is penalized.
- The chart matters a little: +0.08 (30B) to +0.11 (235B) over seeing only the order.
- Earlier visits add nothing over the current note. The task is not longitudinal in practice.
- The agent (9.0 actions, 33k tokens per episode) scores slightly below a single-turn call that is
  handed the current note. The interaction gap is about 0.04.
- A model eight times larger gains 0.04. Inspected answers are clinically sensible and still score
  0.36–0.67, because the score is overlap with the concepts one LLM chose to mention when it wrote
  the reference. Even knowing the encounter's true diagnosis scores 0.21. Most of the distance to 1.0
  is the reference author's wording, which a model could only close by imitating that style.

## Consequence

Across all four official tasks, operating the chart is not what limits score. That is a finding about
the benchmark in its own right and belongs in the write-up.

`patient_diagnosis` remains the best available task: it has a clean reward with no padding exploit,
a large measurable gap (0.42–0.46 for base agents against 0.86 for copy-and-code), and a sub-problem
(final-visit diagnosis, recall about 0.2) that copying cannot solve.
