# Head to head: the 4B with the code book against large models without it

Date: 2026-10-07. Reference arm: tool-SFT `Qwen/Qwen3-4B-Instruct-2507` with the `icd10_search` tool
(milestone 4), public split, seed 0. Comparators: Kimi K2.5, GPT-6 Luna and Gemma 3 27B, untrained and
without the tool, in the same clean `/env` harness: paper prompt, 10 EHR tools, 40-action budget, 50
turns, all 200 `public` patients, one sample per patient. The question, settings and decision rule
were logged before any comparator call (`LOG.md`, 2026-10-07). Raw outputs are in
`results/raw/training/` and `results/raw/audit/`; `heldout` stays sealed.

## Answer

1. **The 4B with the code book beats GPT-6 Luna and is level with Kimi K2.5.** Public strict F1:
   4B tool-SFT **0.738**, Kimi **0.726**, Luna **0.689**. Paired on mean score, the 4B beats Luna by
   **+0.072** (CI +0.034 to +0.111). Against Kimi it is **+0.034** (CI −0.013 to +0.080):
   inconclusive under the rule, but the data rule out Kimi being more than 0.013 ahead.
2. **Without the tool the same 4B clearly trails both.** 8-epoch SFT − Kimi **−0.173**; − Luna
   **−0.134**. The tool is what moves the small model from behind to level.
3. **The two sides win different halves.** The 4B gets **85.4%** of earlier-visit diagnoses right
   against about **70%** for both frontier models. The frontier models win the final visit,
   **58.1–65.4%** against **41.2%**. The code book closes a knowledge gap (coding diagnoses the chart
   already documents), not a reasoning gap (inferring the final visit's new diagnosis).
4. **Gemma 3 27B, the paper's smallest model, scores 0.262** in the same harness, a lower bound because
   its provider returned 46% of its replies empty. The paper's own single-turn code reproduces its
   published number: **0.283** against the paper's **0.287**.
5. **Upstream finding:** the released `eval.cli score` does not attach the chart-neutral sets, so its
   "neutral" F1 for that run is **0.233**, not 0.283.

## Results

Public split, 200 patients, one sample each. Mean score is the mean per-patient official reward
(failures score 0). Strict F1 is the paper's batch F1 with failed sessions counted as zero.

| Arm | Mean score (95% CI) | Strict F1 | Valid submissions | EHR actions | Tokens / patient | Cost |
|---|--:|--:|--:|--:|--:|--:|
| **4B tool-SFT + tool** (milestone 4) | **0.726** (0.689–0.764) | **0.738** | 100% | 3.0 | 12.6k | self-hosted |
| 4B SFT, 8 epochs, no tool (milestone 4) | 0.520 (0.477–0.564) | 0.532 | 99.5% | 3.0 | 7.6k | self-hosted |
| Kimi K2.5, thinking on | 0.693 (0.656–0.730) | 0.726 | 100% | 29.0 | 54.8k | ~$10.2 |
| GPT-6 Luna, reasoning medium | 0.654 (0.620–0.691) | 0.689 | 100% | 15.0 | 10.8k | ~$0.35 |
| Gemma 3 27B, agent (lower bound) | 0.235 (0.198–0.276) | 0.262 | 96.5% | 4.3 | 137.7k | ~$2.2 |
| *Gemma 3 27B, paper's single-turn code (our run)* | *0.268* | *0.283* | *100%* | — | — | *~$0.07* |
| *Gemma 3 27B, paper Table 3* | — | *0.287* | — | — | — | — |

Tokens are provider totals, with history counted on every request. Costs apply list prices
(OpenRouter; DeepInfra for Gemma) to the tokens each run used; they exclude smoke tests. The 4B
scored all 200 patients in 172 s on one rented A100 at concurrency 32. Kimi exhausted the action
budget in 19.5% of episodes and Gemma in 22.6%; both were still forced to a submission. Kimi and
Luna had zero provider errors; Gemma had one. The single-turn row is the paper's batch F1, which
equals strict F1 here because all 200 outputs parsed.

| Paired contrast (mean score) | Difference (95% CI) | Wins / losses / ties | Verdict |
|---|---|---|---|
| 4B tool-SFT − GPT-6 Luna | +0.072 (+0.034 to +0.111) | 107 / 47 / 46 | beats |
| 4B tool-SFT − Kimi K2.5 | +0.034 (−0.013 to +0.080) | 87 / 67 / 46 | inconclusive |
| 4B tool-SFT − Gemma 3 27B | +0.491 (+0.438 to +0.542) | 172 / 13 / 15 | beats |
| 4B SFT 8 epochs − Kimi K2.5 | −0.173 (−0.219 to −0.117) | 48 / 123 / 29 | trails |
| 4B SFT 8 epochs − GPT-6 Luna | −0.134 (−0.181 to −0.081) | 62 / 105 / 33 | trails |
| Kimi K2.5 − GPT-6 Luna | +0.038 (+0.005 to +0.071) | 81 / 46 / 73 | beats |

Paired patient bootstrap, 1,000 resamples, seed 0 (`training.compare`). Decision rule, written
beforehand: "beats" if the CI's lower bound is above 0, "trails" if its upper bound is below 0,
"matches" if the whole CI lies within ±0.05, otherwise "inconclusive". The 4B − Kimi interval
crosses 0 and extends past +0.05, so it is neither a win nor a match.

## The comparators

All three run as agents through the same harness as the 4B, with no code-book tool and no
fine-tuning. Provider routing is enforced in each request with no fallbacks; traces do not store
OpenRouter generation ids, so routing cannot be checked after the fact.

| Model | Why | Settings | Config |
|---|---|---|---|
| `moonshotai/kimi-k2.5` | The paper's best single-turn model (0.732) | Thinking on; no temperature sent, as the paper's adapter does; 32,768 max tokens; SiliconFlow, int4 (its official native precision) | `eval_public_kimi_k25.toml` |
| `openai/gpt-6-luna` | A current budget frontier model | Reasoning effort medium; no temperature (not accepted); 32,768 max tokens; OpenAI standard endpoint | `eval_public_gpt6_luna.toml` |
| `google/gemma-3-27b-it` | The paper's smallest model (0.287) | Temperature 0; 16,384 max tokens; DeepInfra fp8, the paper's registry provider, called directly with our own key through a tool-call translator | `eval_public_gemma3_27b.toml` |

Kimi's 0.726 is close to its paper single-turn 0.732. That is a consistency check, not a validation
of the harness.

## Where each wins

Share of reference diagnoses whose scorer category was submitted (from the `-breakdown.json` files,
`gold_outcomes_by_visit`). Cross-model "named but miscoded" splits are not used, because frontier
models paraphrase diagnosis names.

| Diagnoses | 4B tool-SFT | Kimi K2.5 | GPT-6 Luna | Gemma 3 27B |
|---|--:|--:|--:|--:|
| Earlier visit, documented in a later note (569) | **85.4%** | 69.8% | 70.3% | 26.7% |
| Earlier visit, code carried by no training patient (82) | 57.3% | **59.8%** | 50.0% | 8.5% |
| Final visit, must be inferred (136) | 41.2% | 58.1% | **65.4%** | 17.6% |

The preregistered prediction was that the 4B wins coding and the frontier models win the final
visit. It held.

- **Knowledge gap.** Earlier-visit diagnoses are written in the chart; scoring them is mostly ICD-10
  coding. Training plus the code book puts the 4B 15 points ahead there. On codes it never saw in
  training, the book gives it about Kimi's coding knowledge (57.3% against 59.8%).
- **Reasoning gap.** A final-visit diagnosis has to be reasoned out from findings, and no lookup helps.
  Both frontier models are 17–24 points ahead of the 4B there.
- Most of this benchmark's score is knowledge, which is why the 4B can be level overall.

## Gemma 3 27B

**Tool-call format.** Through OpenRouter's shared DeepInfra pool the smoke test failed on upstream
429s. With our own DeepInfra key, Gemma never issued a structured tool call: it writes prose and then
its native pythonic syntax (```` ```tool_code [open_chart(patient_id=1690)] ``` ````), which DeepInfra
returns as plain text. The harness only nudged, and the model invented results it never received.

**Translator.** `training/gemma_toolcall_proxy.py` is a local OpenAI-compatible proxy in front of
DeepInfra. It does what vLLM's "pythonic" parser does for self-hosted Gemma: when a reply has no
structured `tool_calls`, it parses calls to tools offered in that request with literal arguments and
rewrites them as structured calls. Anything else stays text. Prompts, tools, history and sampling are
unchanged; it also retries 429s. Tests: `tests/test_gemma_toolcall_proxy.py`.

**Lower bound.** The proxy logged 4,510 responses: 600 converted by the translator, 598 structured
calls from DeepInfra's own parser, 1,234 plain text and **2,078 empty (46%)**. In the empty ones
DeepInfra reports about 100 completion tokens but returns neither content nor tool calls: its own
tool parser swallows the output. `tool_choice="none"` would bypass that parser, but it drops the tool
definitions from Gemma's prompt, so it was rejected. Empty turns count against the turn budget and
re-send the whole context, which also inflates the token count and cost. Read 0.262 as a lower bound
for a better-served Gemma.

**Single-turn reproduction.** `audit/paper_singleturn.py` runs the paper's own pipeline inside the
simulator container, unmodified: `eval.runner.run_evaluation` with the locked `cot` strategy, its
parser and its scorer. Only the registry endpoint changes, to DeepInfra direct; temperature 0,
16,384 max tokens and the 120 s timeout are the paper's. All 200 public outputs parsed.

| Gemma 3 27B, public | Value |
|---|--:|
| Paper, Table 3 | 0.287 |
| Paper's code, chart-neutral F1 (our run) | **0.283** |
| Paper's code, released `eval.cli score` | 0.233 |
| Mean per-patient reward | 0.268 |
| Same harness as the 4B, agent (strict F1) | 0.262 |

The agent result and the single-turn result agree, so Gemma's low score is mostly the model, not the
serving. The image lacked `icd10cm_2025.csv`, which upstream's ICD-10 validator needs; it was created
with upstream's own `s05_ontology.download_icd10()`, with no code change.

## Upstream finding: the released scorer drops the chart-neutral rule

The released batch path (`eval.cli score`, `runner._score_run`) never attaches the chart-neutral sets.
Its "neutral" F1 therefore equals the plain F1: **0.233** for the Gemma run, where the chart-neutral
rule gives **0.283**. Upstream's README describes the reported metric as chart-neutral, and the paper's
0.287 matches the rule as the per-episode `score_one` applies it. Our reports use that per-episode
scorer, so every number here and in earlier milestones is chart-neutral. Evidence:
`results/raw/audit/paper_singleturn_gemma3_public.json`.

## Caveats

- **Specialist against generalists, by design.** The 4B was fine-tuned on 680 of the benchmark's
  training patients and given the code book; the comparators were used off the shelf without either.
  This compares a trained specialist with zero-shot generalists, not model capability in general.
- **One sample per patient.** The intervals resample patients, not sampling seeds or training seeds.
- **Provider and precision.** Kimi ran at int4 on one pinned provider (its official precision) and
  Gemma at fp8 on DeepInfra; neither was served at BF16. Routing is enforced by the request and not
  verifiable from traces. Gemma's agent score is a lower bound because of DeepInfra's empty replies.
- **Different sampling controls per arm.** The 4B uses the paper's temperature 0.7 / top-p 0.8 and
  2,048 tokens per turn; Kimi and Luna use provider defaults and 32,768 tokens; Gemma uses
  temperature 0 and 16,384, as in the paper's registry.
- **Stronger frontier models were not run.** Kimi is the paper's best single-turn model and Luna a
  budget model; larger current models would probably score higher.
- **Leaky-harness numbers are not comparable.** The paper's agent scores (0.81–0.89) come from a
  harness that shows the answer key, a known issue listed in upstream's README since its first public
  commit; we measured its size (`results/raw/audit/harness_gap_dev.json`). All arms here use `/env`.
