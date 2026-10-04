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

The final agent comparison should include:

* Base model with a strong fixed prompt
* Prompt-optimized model
* Supervised fine-tuning / behavior cloning
* RL fine-tuning

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

Keep final evaluation on the official metric even if training requires a modified reward.

---

## Phase 3 — Establish learning baselines

Build the agent baselines before serious RL:

### Prompting

Develop a strong fixed prompt and, if useful, run prompt optimization such as GEPA.

### SFT / behavior cloning

Train on successful tool-use trajectories.

Trajectories can come from:

* successful base-model runs
* rejection sampling
* a stronger open model

The goal is to establish whether ordinary imitation learning already captures most of the available improvement.

If the base agent rarely completes episodes successfully, use SFT as a warm start.

---

## Phase 4 — RL

Train on the training patients and select checkpoints on dev.

GRPO is a reasonable starting point, but do not lock in the training framework before a small integration spike. Try the existing Harbor/Verifiers/prime-rl route and a direct `/env` integration, then use whichever is simpler and reliable.

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

---

## Evaluation

Primary metric: official severity-weighted diagnosis F1.

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

1. Does RL outperform a strong prompted agent?
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
* If the reward is genuinely exploitable, fix the training objective while preserving official evaluation.
* If the base agent cannot generate useful trajectories, warm-start with SFT.
* If SFT matches RL, do not manufacture an RL claim.
* If RL raises F1 without meaningful behavioral improvement, investigate whether it learned output priors rather than better chart use.
* If agentic RL becomes impractical, use single-turn RL only as a narrower fallback result.

---

## First implementation milestone

Do **not** build the full RL trainer yet.

First produce:

| Model            | Agent F1 | Valid submit % | Actions | Tokens | Budget exhausted |
| ---------------- | -------: | -------------: | ------: | -----: | ---------------: |
| Qwen candidate A |          |                |         |        |                  |
| Qwen candidate B |          |                |         |        |                  |

And a second table containing the simple task/reward-audit strategies and their scores.

Once those numbers exist, decide whether the benchmark has enough headroom and whether the agent pipeline is healthy enough to justify RL.
