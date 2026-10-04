# Milestone 1 — benchmark verification, task audit, first agent baselines

Date: 2026-09-30. Synthetic Hospital v1.3 at commit `911f34c`, `patient_diagnosis`, `/env` route.
All numbers are on the 120-patient dev split unless stated. Full tables: [tables.md](tables.md).
Raw outputs: `raw/audit/verify.json`, `raw/audit/strategies_{dev,train}.json`, `raw/baselines/*.json`;
full trajectories are in `runs/` (not committed). Code: `audit/` and `baselines/`.

## Status against the milestone

| Item | Status |
|---|---|
| Environment and official scorer running end to end | Done |
| Oracle scores correctly | Done: 1.000 on all 1,000 train + public instances |
| Malformed answers fail cleanly | **Partly.** Two malformed shapes crash the scorer (below) |
| Observations do not expose hidden labels | **Mostly.** No reference codes exposed; one filter hole (below) |
| How prior diagnoses appear in later encounters | Done: this is the main finding |
| Dev split, held-out sealed | Done: 680 train / 120 dev; `heldout` unreachable from the CLI |
| Agent baseline table | **Partly.** Two reference models run; the primary model is not run yet |
| Task/reward audit table | Done |

The primary model, `Qwen/Qwen3-4B-Instruct-2507`, has no baseline yet. OpenRouter does not serve
it (nor Qwen3.5-4B), so it needs a vLLM endpoint on a GPU. The two models below are reference
points served by OpenRouter, not the candidates named in the plan.

## Table 1 — agent baselines (paper prompt, 40-action budget, temperature 0.7, one rollout per patient)

| Model | n | Mean reward (agent F1) | 95% CI | Official F1 (strict) | Precision | Weighted recall | Valid submit % | Actions | Tokens / episode | Budget exhausted % |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| Qwen3.5-9B, reasoning off (DeepInfra/Parasail, bf16) | 120 | 0.415 | 0.344–0.483 | 0.441 | 0.514 | 0.410 | 96.7 | 16.2 | 118,804 | 5.8 |
| Qwen3-30B-A3B-Instruct-2507 (provider not pinned) | 120 | 0.463 | 0.405–0.525 | 0.494 | 0.544 | 0.467 | 98.3 | 11.2 | 59,280 | 0.0 |

Tokens are prompt plus completion summed over turns (the context is re-sent every turn).
Caveats: one sampled rollout per patient; the 30B run was not pinned to a provider, so
quantization may vary between requests; Qwen3.5-9B with default reasoning on was started and
abandoned at 55 of 120 episodes because it was about ten times slower and off-target for a
no-think project (partial trajectories are in `runs/qwen_qwen3.5-9b-dev/`, not reported).

The pipeline is healthy for both: every episode ended in a submission, no malformed submissions,
almost no tool-argument errors.

## Table 2 — task/reward audit (hand-written strategies, official scorer)

| Strategy | Mean reward | Precision | Weighted recall | # predictions |
|---|--:|--:|--:|--:|
| Oracle | 1.000 | 1.000 | 1.000 | 3.5 |
| Empty or malformed submission | 0.000 | — | — | 0 |
| Profile problem list (`view_problem_list`), coded by lookup | 0.138 | 0.902 | 0.162 | 1.3 |
| **Note "Active Problem List" entries with a "(diagnosed date)", coded by lookup** | **0.859** | 0.985 | 0.787 | 2.9 |
| Note "Active Problem List", all entries, coded by lookup | 0.861 | 0.976 | 0.795 | 3.7 |
| Every diagnosis name mentioned anywhere in the chart | 0.412 | 0.298 | 0.859 | 16.5 |
| Top-5 / top-20 / top-100 training categories | 0.029 / 0.021 / 0.017 | ≤0.03 | 0.03–0.23 | 5 / 20 / 100 |
| The scorer's 11 "critical" categories | 0.012 | 0.008 | 0.028 | 11 |
| Oracle + 5 / + 20 wrong codes | 0.518 / 0.236 | 0.367 / 0.140 | 1.000 | 8.5 / 23.5 |
| Oracle + a code in every chart-neutral category (n=40) | 1.000 | 1.000 | 1.000 | 10.0 |
| Oracle with acuity flipped | 1.000 | 1.000 | 1.000 | 3.5 |
| Oracle truncated to 3-character categories | 0.921 | 0.927 | 0.915 | 3.5 |
| Oracle with every entry duplicated | 0.748 | 0.608 | 1.000 | 7.0 |
| Oracle minus diagnoses of the final encounter | 0.880 | 1.000 | 0.810 | 2.9 |

On the 680 training patients the problem-list copy strategy scores 0.818 (95% CI 0.804–0.832)
and "oracle minus final encounter" 0.845.

"Coded by lookup" means the extracted name is mapped to a code by exact match in the benchmark's
own diagnosis table. That is an upper bound on "copy the name and code it correctly"; a model has
to produce the code itself.

## Findings

### 1. Most of the task is copying a list that the chart already contains

Each later note's PMH section carries an "Active Problem List" in which the reference diagnoses
of earlier encounters appear verbatim, tagged "(diagnosed YYYY-MM-DD)". 81% of reference
diagnosis names are readable this way; upstream documents this as by construction. Only the
diagnoses established at the patient's final encounter (19%) have to be inferred from findings,
and only 13% of those are named in their own note.

Copying the dated entries and coding them correctly scores 0.86. The ceiling for a policy that
never diagnoses anything is 0.88. Genuine diagnostic reasoning is worth at most the last ~0.12.

### 2. Current agents are far below the copy baseline, and the gap is not chart navigation

Both models read most of the chart (encounter coverage 0.94 and 0.80; the 30B model opens
the final encounter in 99% of episodes), yet they recover only about half of the diagnoses that are
written out for them, and about 20% of final-encounter diagnoses.

Of the copyable reference diagnoses each model missed or hit:

| | Qwen3.5-9B | Qwen3-30B |
|---|--:|--:|
| Correct category submitted | 48% | 51% |
| Named correctly, wrong ICD-10 category | 27% | 17% |
| Name was in the observations, not submitted | 24% | 32% |
| Name never observed | 0.3% | 0% |

So the available improvement is dominated by (a) transcribing the dated problem-list entries
and (b) recalling the right ICD-10-CM category for a known name, plus (c) precision: the models
submit 6–7 entries against 3.5 in the reference. None of these is better chart use.

### 3. What the reward is and is not sensitive to

- Predicted acuity has no effect on the reward. Acuity accuracy is reported separately.
- Which list an entry is in (`active_diagnoses` vs `chronic_conditions`) has no effect.
- Extra predictions in the patient's chart-neutral categories are free (mean 6.3 such categories
  per patient among those probed, up to 17). Extra predictions elsewhere are penalized hard:
  five wrong codes halve the reward. Pure padding and prior-guessing strategies score under 0.03,
  so there is no cheap exploit of that kind.
- Matching is at the 3-character category, except injuries and poisonings (S, T), which need 5.
- Duplicated entries are penalized as wrong predictions.

### 4. Upstream defects

- **Scorer crash on some malformed submissions.** A submission whose lists contain strings
  (`{"active_diagnoses": ["I10"], ...}`) or are strings raises HTTP 500 instead of scoring 0, and
  through `/env/step` the episode is left open. The README says malformed submissions score 0
  and never raise. `baselines/agent.py` treats this case as a malformed submission with reward 0,
  and the training environment validates arguments before they reach the scorer. Any RL setup
  must do one of the two, or it will crash on a model's bad output.
- **Assessment/plan filter hole.** `view_section(section_id)` returns assessment and plan
  sections that every other tool strips; section ids are contiguous, so the hidden ones are
  guessable. In all 34 direct reads tried, none was blocked. Practical value looks low: these
  sections hold question-stem residue, and no reference name that was otherwise unreadable
  appeared in them. The 30B model called `view_section` 32 times in 120 episodes. Worth watching
  during RL, and worth reporting upstream with the crash.
- **The reward depends on a boot-time download.** Chart-neutral sets fall back to fuzzy matching
  against ICD-10-CM titles that the container fetches from CMS on first boot. If that download
  fails, rewards differ. `python -m audit.verify` checks that the table is populated (97,584 rows here).
- **The official aggregate drops empty submissions** from both means instead of counting them
  as zero, so an agent that fails to submit is not penalized by it. `official_f1_strict` in this
  repo counts them as zero. The paper prompt also has a templating slip ("through You have
  access to...") that every `/env` episode inherits.

## Reading against the decision rules

- *"If the task is mostly solved by trivial extraction, reassess before investing heavily."*
  This applies. A regex plus a correct name-to-code table reaches 0.86, and base agents sit at
  0.42–0.46, so there is headroom, but the headroom is mostly transcription and ICD-10 recall.
  SFT or RL will very likely show a large F1 gain, and that gain will mostly be "learned to copy
  the problem list and code it", which is an output-prior result rather than a chart-use result.
- *"If the reward is genuinely exploitable, fix the training objective."* No padding or prior
  exploit exists. The chart-neutral waiver and the acuity blind spot are loopholes worth logging
  during training, not reasons to change the reward yet.
- *"If the base agent cannot generate useful trajectories, warm-start with SFT."* Not needed on
  the evidence so far: the reference models complete episodes. Unknown for the 4B model.

## Open decisions

1. **How to frame the research question given finding 1.** Options: (a) keep the official task
   and report the copy baseline as the bar every method must clear; (b) add a harder secondary
   metric restricted to final-encounter diagnoses, where copying scores zero, and analyse RL
   gains separately on it; (c) train on a variant that masks the dated problem-list entries.
   Recommendation: (a) plus (b). The official metric stays the headline, and the
   final-encounter recall isolates the part of the task that needs reasoning over evidence.
2. **Where to serve Qwen3-4B-Instruct-2507** for the candidate table (vLLM on a rented GPU, which
   training needs anyway, or a quantized local copy for a quick viability read).
