# Synthetic Hospital RL — Project Plan

## Goal

Train a small open-weight model to diagnose patients by operating Synthetic Hospital’s longitudinal EHR environment, and determine what **RL adds over strong prompting and supervised fine-tuning**.

The core question is:

> **Can RL teach a small model to use a realistic longitudinal medical record environment more effectively, and do those gains generalize beyond the training cases?**

This is a portfolio project. The valuable result is not “we ran GRPO,” but a credible analysis of **what training changed, where it helped, and whether the model learned better chart use rather than benchmark shortcuts.**

Focus on the `patient_diagnosis` task and the corrected self-retrieving `/env` or Harbor environment. Do not use the original agent harness with known label leakage.

Use a Qwen-family instruct model around 4–8B.

### Model choice (decided 2026-09-30)

| Candidate | Role |
| --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | Primary / expected choice |
| Qwen3.5-4B | Sanity-check whether a newer 4B materially changes baseline viability |
| Qwen3.5-9B | Escalation only if 4B models cannot operate the environment |

Rationale: Prime-RL ships an end-to-end multi-turn tool-use RL example (Wiki Search) using exactly Qwen3-4B-Instruct-2507 with Verifiers, LoRA, vLLM, and native function calling. The model is a 4B dense text-only causal LM, Apache 2.0, 262K context, improved tool use, no `<think>` mode. Qwen3.5-4B is multimodal, which adds stack novelty with no benefit here.

Do not use OpenRouter's generic `qwen/qwen3-4b` as the experimental model: it is the original April 2025 Qwen3-4B, not the 2507 instruct checkpoint. Train and evaluate the exact Hugging Face checkpoint.

---

## Main comparison

The active checkpoint comparison uses the **unchanged paper prompt** throughout:

* Exact untrained `Qwen/Qwen3-4B-Instruct-2507`
* Supervised fine-tuning / behavior cloning
* RL fine-tuning from the SFT checkpoint

Prompt engineering is a completed negative control, not an ongoing branch.
Repeated strong-prompt comparisons found no demonstrated gain. Retain their
artifacts, but do not run strong-prompt or prompt-optimization experiments after
new checkpoints. This paper-only decision was confirmed on 2026-10-04.

Frontier models and published benchmark results are reference points, not the main comparison.

Single-turn diagnosis is a **diagnostic and fallback path**, not a second full research project. Use it to debug the task, test model capability, or warm-start training if useful.

---

## Phase 1 — Verify the benchmark

Before training:

* Get the environment and official scorer working end-to-end.
* Confirm an oracle answer scores correctly and malformed answers fail cleanly.
* Verify that observations available to the policy do not expose hidden labels.
* Inspect how prior diagnoses and problem lists appear in later encounters.
* Reproduce at least one reasonable open-model baseline.
* Create a dev split from the training patients.
* Treat held-out patients as sealed final evaluation data.

Run a few candidate Qwen models in the agent environment and measure:

* diagnosis F1
* valid submission rate
* actions
* tokens
* budget exhaustion

Pick a model with enough capability to operate the environment but enough headroom to improve.

---

## Phase 2 — Understand the task and reward

Before RL, test simple hand-written strategies such as:

* copying the latest problem list
* emitting all chart-mentioned diagnoses
* predicting common diagnosis categories
* padding with extra diagnoses
* guessing high-severity conditions
* perturbing acuity
* malformed or empty answers

These serve two purposes:

1. detect genuine reward exploits
2. establish how much of the task can be solved through simple extraction

Do not automatically classify a high-scoring extraction strategy as an exploit. If copying documented diagnoses legitimately solves much of the benchmark, that is an important property of the task.

Keep the official reward unchanged for both training and evaluation.

---

## Phase 3 — Establish learning baselines

Build the agent baselines before serious RL:

### Prompting — complete

The paper and strong-prompt comparisons are recorded in `LOG.md`. Prompting did
not produce a demonstrated improvement and is retired from the active plan.

### SFT / behavior cloning

The completed control trained on 680 label-informed `oracle-copy-v1`
demonstrations, with no dev or heldout examples used for gradients. These are
reference-answer demonstrations, not trajectories from an independently
successful policy; final-visit labels can go beyond what the note reveals.

Three epochs of LoRA SFT increased paper-prompt dev mean official reward from
**0.1017 to 0.3961**. Finalized episodes used three EHR actions without nudges or
forced submission. Earlier-visit recall improved much more than final-visit
recall. This checkpoint is the control for measuring what RL adds.

---

## Phase 4 — RL

Train only on the 680 training patients, starting from the completed SFT
checkpoint. First verify real rollout tokens/log-probabilities, assistant-only
loss masking, official reward delivery, and policy-weight refresh in Prime-RL.
The two-step setup check is not an experimental result and its adapter is not
the starting point for the full run.

Initial controlled arm: group-relative advantages with a PPO-clipped objective,
LoRA rank 16 / alpha 32 / dropout 0, learning rate 1e-5, four rollouts per patient,
32 rollouts per update, and 85 updates. This is one nominal pass over the training
patients; record actual coverage and errored/stale-rollout exclusions. Use the
paper prompt, unchanged official reward, and no length penalty or added reward.

Evaluate the predeclared terminal checkpoint once on all 120 dev patients with
the same paper-prompt evaluation stack and controls as SFT. Do not choose the
best-looking intermediate dev result. Heldout remains sealed.

Prediction: a smaller gain than SFT, possibly no statistically clear gain.
Protocol compliance is already largely repaired; any gain needs diagnosis/code
and seen-versus-unseen analysis before attributing it to medical reasoning.

This is a pilot of one RL recipe against one SFT recipe, not a claim that either
method is exhausted. After the pilot, assess SFT train-versus-dev performance
and define a continued-SFT control from the same preserved checkpoint, with a
comparable additional GPU budget and the same prompt/data split. A method-level
claim for RL requires separating the effect of the objective from the effect of
more training. Do not launch an open-ended SFT hyperparameter sweep or treat
repeated dev selection as an unbiased final test.

During training, log full trajectories and continuously inspect:

* reward
* official F1
* valid completion rate
* tool usage
* trajectory length
* token usage
* budget exhaustion
* suspicious prediction patterns

The interesting question is not simply whether reward rises, but **what behavior changes as it rises**.

### Outcome (2026-10-05; details in `results/milestone3_training.md`)

The pilot ran as declared and its prediction held. On dev (seed 0) RL scored
0.406 versus SFT 0.396 (+0.010, CI −0.019 to +0.038); five sampling seeds
gave +0.020 (+0.003 to +0.038). The untouched public split agreed: RL − SFT
was +0.008 (−0.014 to +0.032). The continued-SFT control (three more epochs of
the same recipe, ~0.6 GPU-h versus RL's ~4.1) beat both SFT and RL on dev and
public (public +0.047 and +0.038). RL did not change behaviour; it sharpened codes the policy already
sampled, while about a third of training diagnoses were never coded correctly
in any of four samples. After SFT the agent names almost every documented
diagnosis; the remaining error is ICD-10 coding, which tracks how often a code
appeared in training. Final-visit diagnoses remain the genuine inference gap.
By the decision rule "if SFT matches RL, do not manufacture an RL claim", this
phase reports no RL advantage.

---

## Evaluation

Primary training/comparison metric: mean official per-episode reward (`weighted_problem_list_f1_neutral`). Also report the paper's batch F1 and its failure-inclusive strict variant.

Also report:

* precision
* recall
* acuity accuracy
* parse/completion failures
* actions per trajectory
* tokens per trajectory
* budget exhaustion

Evaluate generalization along at least two dimensions:

### Seen vs. unseen diagnoses

Separate diagnosis categories that appeared in training from those that did not.

Treat improvements on unseen diagnoses as evidence of transfer, not proof that the model learned new medical knowledge.

### Chart complexity

Analyze performance by chart length, encounter count, and related measures of case complexity.

Control for obvious confounds where practical; longer charts may correspond to systematically different patient types.

For the final models, inspect trajectories qualitatively:

* What information does the model retrieve?
* Does it search more selectively?
* Does it stop earlier?
* Does it integrate evidence across encounters?
* Are gains mostly coming from better answer priors or better chart interaction?

Use bootstrap confidence intervals where practical and multiple training seeds if compute permits.

---

## Final questions the project must answer

1. Does RL outperform the untrained paper-prompt agent?
2. Does RL outperform SFT / behavior cloning?
3. Where do the improvements occur?
4. Does agent behavior meaningfully change?
5. Does performance hold on unseen diagnoses and harder charts?
6. What simple strategies already solve part of the benchmark?
7. What reward exploits or failure modes emerge?

If RL does not beat SFT, report that plainly. That is still a legitimate result.

---

## Deliverables

Release:

* training and evaluation code
* configs
* fixed train/dev split
* model adapter or weights
* cached evaluation outputs
* training curves
* representative trajectories
* reward/task audit results
* compute cost
* technical blog post
* concise thread summarizing the result

The write-up should emphasize **what the model learned and what the benchmark actually measures**, not just leaderboard performance.

---

## Decision rules

* If the task is mostly solved by trivial extraction, reassess whether agentic RL can produce an interesting result before investing heavily.
* If the reward is genuinely exploitable, stop and report the limitation rather than silently changing the agreed official objective.
* If the base agent cannot generate useful trajectories, warm-start with SFT.
* If SFT matches RL, do not manufacture an RL claim.
* If RL raises F1 without meaningful behavioral improvement, investigate whether it learned output priors rather than better chart use.
* If agentic RL becomes impractical, use single-turn RL only as a narrower fallback result.

---

## Current implementation milestone

Benchmark audit, prompting baselines, SFT, the SFT→RL pilot, the continued-SFT
control, the public final test, the SFT epoch sweep and the ICD-10-CM code-book
tool are complete; see `LOG.md`, `results/milestone3_training.md`,
`results/milestone4_sft_and_codebook.md` and `results/raw/training/`. Heldout remains sealed.

Best result: 8-epoch SFT on demonstrations that look codes up in the public code
book, public mean reward 0.726, versus 0.520 for 8-epoch SFT without it. Next, if
continued: write-up and release, the authors heads-up and upstream bug reports.
The open training question is final-visit inference: 57% of those diagnoses are
still never named. A preregistered sampling test (LOG.md, 2026-10-07) found that
RL from tool-SFT on its own 680 training patients has almost no signal (78.5% of
four-sample groups score identically). An RL arm needs patients the SFT never
saw, such as a split of the 680, and on dev its learnable signal is mostly
final-visit and weaker than the first RL run's. Not planned unless requested.
