# What RL adds over SFT on `patient_diagnosis`

Date: 2026-10-05. Model: exact `Qwen/Qwen3-4B-Instruct-2507`, LoRA rank 16. Every arm uses the
unchanged paper prompt, the 40-action `/env` harness, temperature 0.7 / top-p 0.8, the official
scorer and the fixed split (680 train / 120 dev). The paper's 200-patient `public` split was held
back as the final test; `heldout` stays sealed. Protocol and every decision, with times, are in
`LOG.md` (2026-10-04 to 2026-10-05). Raw outputs are in `results/raw/training/`.

## Answer

RL did not add a meaningful gain over SFT. One more pass of plain SFT did, at a seventh of the
compute. The remaining error is ICD-10 coding of diagnoses the agent already reads off the chart,
which neither method fixes for diagnoses it has not seen in training.

| Arm (dev, seed 0) | Mean reward | 95% CI | Training compute |
|---|--:|---|---|
| Untrained 4B | 0.102 | 0.072–0.136 | — |
| SFT, 3 epochs on 680 oracle-copy demonstrations | 0.396 | 0.331–0.463 | ~0.6 GPU-h |
| SFT → RL (GRPO, 85 updates, 32 rollouts each) | 0.406 | 0.338–0.473 | ~4.1 GPU-h more |
| SFT → 3 more SFT epochs (continued SFT) | **0.477** | 0.414–0.538 | ~0.6 GPU-h more |

| Paired contrast (dev) | Seed 0 | Five sampling seeds per arm |
|---|---|---|
| RL − SFT | +0.010 (−0.019 to +0.038) | +0.020 (+0.003 to +0.038) |
| Continued SFT − SFT | +0.081 (+0.034 to +0.127) | +0.083 (+0.043 to +0.124) |
| Continued SFT − RL | +0.071 (+0.021 to +0.122) | +0.063 (+0.023 to +0.106) |

Paired patient bootstrap, 1,000 resamples; failures stay in at zero. The seed-0 contrast was the
preregistered primary; five-seed means (seeds 0–4) were a preregistered secondary.

## Final test: the paper's public split (200 patients, untouched until these four arms were fixed)

| Arm (public, seed 0) | Mean reward | 95% CI | Official F1 |
|---|--:|---|--:|
| Untrained 4B | 0.158 | 0.120–0.198 | 0.177 |
| SFT | 0.379 | 0.331–0.429 | 0.388 |
| SFT → RL | 0.387 | 0.340–0.433 | 0.398 |
| Continued SFT | **0.425** | 0.381–0.468 | **0.436** |
| *Reference: copy dated problem-list entries, coded with the benchmark's table* | *0.804* | *0.771–0.831* | *0.828* |

| Paired contrast (public) | Difference | 95% CI | Wins / losses |
|---|--:|---|--:|
| SFT − untrained | +0.221 | +0.183 to +0.259 | 114 / 14 |
| RL − SFT | +0.008 | −0.014 to +0.032 | 30 / 34 |
| Continued SFT − SFT | +0.047 | +0.018 to +0.079 | 50 / 32 |
| Continued SFT − RL | +0.038 | +0.010 to +0.068 | 51 / 36 |

Paper reference points on the public split (its own harnesses, not ours): the best single-turn
model scores severity-weighted F1 0.73 (Table 3), level with the mean of seven physicians on a
13-patient subset (best physician 0.89). Its self-retrieving agents score **0.81–0.89** on 100
public patients (Table 4: GPT 5.3 0.84, Mistral Large 0.81, Llama 4 Scout 0.89). Our harness gave
Qwen3-30B only 0.47 on dev, so that gap is unexplained and harness differences (13 versus 10 tools,
prompting) must be resolved before comparing our agents with the paper's. The paper reports no
fine-tuning or RL results. Every public conclusion here matches dev.

## Where the error is: coding, not chart reading

Each reference diagnosis is either a *hit* (its scorer match key was submitted), *named but
miscoded* (its exact name was submitted under another code), or *not named*.

| Dev, diagnoses first seen at an earlier visit (347 of 417) | Hit | Named, miscoded | Not named |
|---|--:|--:|--:|
| Untrained | 15.3% | 25.9% | 58.8% |
| SFT | 49.0% | 50.7% | 0.3% |
| RL | 50.1% | 49.9% | 0.0% |
| Continued SFT | 55.0% | 44.4% | 0.6% |

After SFT the agent writes down essentially every diagnosis the chart already documents and
mis-codes half of them. Keeping the agent's own names and swapping in the benchmark's code for
each name (`training.recode`, a diagnostic counterfactual, not an agent score) moves SFT from
0.396 to **0.821** and RL from 0.406 to 0.821 on dev, close to the copy-the-problem-list ceiling
of 0.859. Public replicates it: SFT 0.379 → 0.778, RL 0.387 → 0.790, continued SFT 0.425 → 0.772,
untrained 0.158 → 0.358 (copy ceiling 0.804); earlier-visit diagnoses there are named 99.5% of
the time after SFT and coded correctly 46%. Final-visit diagnoses are the real inference gap:
76–80% are never named on dev, 64–69% on public.

Coding accuracy follows training exposure (earlier-visit diagnoses the model named):

| Training patients carrying the code | Untrained | SFT | RL | Continued SFT |
|---|--:|--:|--:|--:|
| 5 or more | 54% | 81% | 82% | 81% |
| 2–4 | 22% | 39% | 41% | 53% |
| 1 | 24% | 20% | 20% | 26% |
| 0 (unseen) | 1/8 | 8% | 8% | 11% |

Wrong codes are confident and often borrowed: 74% of SFT's mis-codes reuse a code from its
training targets (Hypomagnesemia → E53.1, vitamin B6 deficiency, instead of E83.42; Interstitial
lung disease → J44.9, COPD, instead of J84.9). Only 20% of the benchmark's diagnosis names are
literal ICD-10-CM titles, so a public code book alone would not supply the mapping.

## Why RL could not fix it

GRPO learns only from codes the policy already samples. Across RL's four-rollout groups on
training patients, a third of earlier-visit diagnoses were never coded correctly in any of the
four samples (no learning signal), a third were mixed, and a third were already always right.
Over training the mixed share fell from 38% to 25% and the always-right share rose from 29% to
40%, while the never-right share stayed at 33–35% (72% for codes that no other training patient
carries). Equal-reward groups rose from 25% to 37%. RL sharpened what SFT sometimes got right; it
did not discover new codes. Its dev scores barely depend on the sampling seed (0.4051–0.4066
across five seeds, versus 0.3552–0.4130 for SFT), and its answers differ from SFT's less than two
SFT samples differ from each other (89 versus 103 of 120 patients).

An exploratory decoding check (dev only, added after the replicates, not part of the protocol)
makes the same point: decoded greedily, SFT scores 0.430 and RL 0.416 (RL − SFT −0.014, −0.050
to +0.018). Turning SFT's temperature down recovers everything RL added at the paper's
temperature; continued SFT stays ahead under greedy decoding too (0.477, +0.047 over SFT).

SFT, by contrast, was underfit: on its own 680 training patients, whose demonstrations submit the
exact answer, it scored 0.482 (RL 0.502). Three more epochs raised that to 0.696 and dev to 0.477,
mostly through codes seen in two to four training patients.

## Behaviour

Every trained arm reads the chart in exactly three actions (encounters list, latest note, submit),
with no nudges and no forced submissions. RL did not change behaviour: same actions, same output
length (193 vs 195 tokens), 3.4 predictions per patient. The untrained model used 8.8 actions, 16
nudges and exhausted the budget in half of its episodes.

## Caveats

- One RL recipe and one continued-SFT recipe, one training seed each. RL hyperparameters were
  fixed in advance and not tuned on dev.
- Continued SFT is a fresh adapter on the merged SFT checkpoint; it is "six epochs" only loosely
  (two cosine schedules).
- Dev was used for several analyses; the public split is the untouched test for these arms only.
- The coding counterfactual uses the benchmark's own diagnosis table; it diagnoses where error
  lives and is not comparable to any agent score.
- RL finalized 5,505 rollouts and trained on 2,720. The orchestrator's discard warnings (71 of 85
  steps) attribute 1,605 discards to staleness after weight updates and 1,503 to equal-reward groups.
