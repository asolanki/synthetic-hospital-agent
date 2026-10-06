# Train longer, then give the model a code book

Date: 2026-10-06. Model: exact `Qwen/Qwen3-4B-Instruct-2507`, LoRA rank 16. Same fixed split (680
train / 120 dev), official scorer and paper prompt as `milestone3_training.md`. The paper's
200-patient `public` split was evaluated once per arm at the end; `heldout` stays sealed. The
protocol, selection rule and every amendment were logged before the results they govern
(`LOG.md`, 2026-10-06). Raw outputs are in `results/raw/training/`.

## Answer

1. **SFT was undertrained.** The unchanged recipe peaks at 8 epochs, not 3: dev **0.576** versus
   0.381, and public **0.520** versus 0.379. It beats RL and continued SFT on both splits.
2. **A public ICD-10-CM code book, offered as one search tool, is worth more than any training we
   tried — once the model is taught to use it.** Tool-SFT, the same demonstrations with one
   lookup turn inserted, scores **0.776** on dev and **0.726** on public. Public official F1 is
   **0.738**, matching the paper's best frontier single-turn model (0.732). Codes never seen in
   training go from **1%** to **57%** correct.
3. **The tool alone helps untrained models**: untrained 4B **0.10 → 0.48** on dev and
   **0.16 → 0.42** on public; Qwen3-30B **0.42 → 0.60** and **0.43 → 0.61**. A model already fine-tuned
   into a fixed routine ignores a tool it was never shown, so the 8-epoch SFT model gained nothing.

## Results

| Arm | Dev (mean of 3 seeds) | Public (seed 0) | Public official F1 |
|---|--:|--:|--:|
| Untrained 4B | 0.102* | 0.158 | 0.177 |
| Untrained 4B + tool | 0.482* | 0.420 | 0.496 |
| SFT, 3 epochs (milestone 3) | 0.381 | 0.379 | 0.388 |
| SFT → RL (milestone 3) | 0.406 | 0.387 | 0.398 |
| Continued SFT (milestone 3) | 0.467 | 0.425 | 0.436 |
| **SFT, 8 epochs** (sweep selection) | **0.576** | **0.520** | 0.534 |
| SFT, 8 epochs + tool (never trained on it) | 0.569 | 0.517 | 0.529 |
| **Tool-SFT, 8 epochs + tool** | **0.776** | **0.726** | **0.738** |
| Qwen3-30B, temperature 0 | 0.420* | 0.427 | 0.461 |
| Qwen3-30B + tool, temperature 0 | 0.599* | 0.614 | 0.678 |
| *Reference: copy dated problem-list entries, coded with the benchmark's table* | — | *0.804* | *0.828* |

\* seed 0 only. Mean reward is the official per-episode score; failures stay in at zero.

| Paired contrast | Dev | Public |
|---|---|---|
| SFT 8 epochs − SFT 3 epochs | +0.195 (+0.138 to +0.256) | +0.141 (+0.099 to +0.186) |
| SFT 8 epochs − RL | +0.170 (+0.113 to +0.230) | +0.133 (+0.091 to +0.174) |
| SFT 8 epochs − continued SFT | +0.109 (+0.061 to +0.158) | +0.095 (+0.060 to +0.133) |
| Tool-SFT + tool − SFT 8 epochs | +0.200 (+0.151 to +0.252) | +0.206 (+0.160 to +0.252) |
| Untrained + tool − untrained | +0.380 (+0.321 to +0.435) | +0.262 (+0.216 to +0.311) |
| Untrained + tool − SFT 3 epochs | +0.086 (+0.003 to +0.167) | +0.041 (−0.014 to +0.099) |
| SFT 8 epochs + tool − SFT 8 epochs | −0.007 (−0.023 to +0.006) | −0.003 (−0.021 to +0.014) |
| Qwen3-30B + tool − Qwen3-30B | +0.179 (+0.115 to +0.251) | +0.187 (+0.136 to +0.235) |

Paired patient bootstrap, 1,000 resamples, seed 0; dev arms with three seeds are averaged per patient first.

## The epoch sweep

One fresh run of the unchanged recipe for 10 epochs on one cosine schedule. Every epoch's adapter
was scored on dev at three sampling seeds. Selection rule, written beforehand: highest three-seed
mean, earliest epoch within 0.005.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 | 10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Dev | 0.247 | 0.326 | 0.445 | 0.484 | 0.549 | 0.546 | 0.570 | **0.576** | 0.567 | 0.565 |
| Own 680 training patients | | | 0.609 | | | 0.887 | | | | 0.986 |

Dev flattens after epoch 8 while the model memorizes its demonstrations: mild overfitting, not a
cliff. Epoch 8 is a mid-schedule checkpoint, so its learning rate was not annealed.

## The tool

`icd10_search(query)` runs BM25 over the public CMS ICD-10-CM 2025 release only: tabular titles,
inclusion terms, the alphabetic index, the Neoplasm Table and the Table of Drugs and Chemicals.
The benchmark's own diagnosis table is never indexed, so the tool cannot reveal a patient's
reference codes. Search changes were made on training-patient names only, then frozen before any
agent saw the tool. Code: `training/environment/synthetic_hospital_dx/icd10.py`, `icd10_tool.py`.
It is an augmented harness, off by default; official-harness numbers are unchanged.

Only 11.5% of reference diagnosis names equal an official ICD-10-CM title, so the agent must
still choose among results: searching a dev reference name returns the right scorer category
first 72.7% of the time and within the top 8 85.1%. Tool-SFT beats the naive "take the top
result" recode (dev 0.67), so it does more than copy the first hit. Its demonstrations keep the
reference code even for the 17% of training lookups whose results miss it.

Where tool-SFT's gain comes from (public, diagnoses first documented at an earlier visit, scorer
category submitted):

| Training patients carrying the code | 8-epoch SFT | Tool-SFT |
|---|--:|--:|
| 5 or more | 84% | 96% |
| 2–4 | 68% | 87% |
| 1 | 35% | 83% |
| 0 (never seen in training) | 1% | **57%** |

Final-visit diagnoses also improve (public recall 0.32 → 0.41); 57% are still never named.
Tool-SFT emits more tokens (about 300 versus 200 output tokens per patient) and its acuity labels
are slightly worse (0.88 versus 0.94; acuity does not enter the official reward).

## Caveats

- **Augmented harness.** The paper's agents do not have this tool; compare tool arms with each
  other, and with the paper's numbers only as context. The paper's own agent scores (0.81–0.89)
  come from a harness that leaks the reference codes (`results/raw/audit/harness_gap_dev.json`).
- **Label-informed demonstrations.** As in milestone 3, SFT clones the benchmark's reference
  answers on training patients, including its naming and coding conventions.
- **Untrained models loop.** Without training, Qwen3-30B sometimes searches until the turn cap:
  11 of 120 dev episodes, scored zero. On public its valid-submission rate fell from 99% to 95%
  (cause not inspected). A lookup cap would likely help; it was not added, to keep the
  preregistered harness fixed.
- One training seed per recipe. Dev chose the epoch count; public was untouched until the end and
  confirms every dev conclusion.
