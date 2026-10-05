# Synthetic Hospital RL

What does RL add over strong prompting and supervised fine-tuning when a small open-weight
model (Qwen3-4B-Instruct-2507) has to diagnose patients by operating a longitudinal EHR?
The plan is in [PLAN.md](PLAN.md), every finding and decision in [LOG.md](LOG.md).

Task: `patient_diagnosis` in [Synthetic Hospital](https://github.com/sparkcpark/synthetic_hospital)
v1.3, played through the simulator's corrected `/env` route (never the original `eval/agents`
harness, which leaks the labels through the problem list).

## How the pieces fit

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ hospital/  the benchmark server (HTTP): patients, answer keys, scorer                │
│ http://localhost:58000                                                               │
└─────────────┬─────────────────────────────┬─────────────────────────────┬────────────┘
              │ HTTP                        │ HTTP                        │ HTTP
┌─────────────┴────────────┐  ┌─────────────┴────────────┐  ┌─────────────┴────────────┐
│ audit/      Phases 1–2   │  │ baselines/  Phase 3a     │  │ training/   Phases 3b–4  │
│ is the benchmark sound,  │  │ first agent harness,     │  │ Verifiers environment    │
│ what does it measure?    │  │ OpenRouter models        │  │ prime-rl evals/trains on │
│ DONE                     │  │ DONE                     │  │ IN PROGRESS              │
└──────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘
```

Shared code lives in `common/` (simulator client, benchmark database reader, split, metrics).
`training/environment/` is the exception: prime-rl loads it by name as a standalone package,
so it imports nothing from `common/`.

One episode: the model gets the paper's instructions and 10 EHR tools, calls tools (each
forwarded to the server, which hides the answer key and counts the 40-action budget), and ends
by calling `submit_diagnosis`. The server scores the submission; that number is the reward.

## Repo map

| Path | What it is | Status |
|---|---|---|
| [`hospital/`](hospital/README.md) | Start the benchmark server | — |
| [`common/`](common/) | Small shared library: simulator client, benchmark DB reader, fixed split, metrics | — |
| [`audit/`](audit/README.md) | Phase 1–2 checks: scorer sanity, label leaks, hand-written strategies, other tasks | done |
| [`baselines/`](baselines/README.md) | First agent harness and the milestone-1 baseline table | done |
| [`training/`](training/README.md) | Verifiers environment, self-hosted 4B inference/eval, LoRA SFT, Prime-RL GRPO, paired comparisons and breakdowns | done (pilot) |
| `data/splits/` | Fixed train/dev split (680 / 120 patients) | committed |
| `data/sft/` | SFT demonstrations | committed |
| `results/` | Write-ups (below), `tables.md`, `cases/`, raw JSON in `raw/<phase>/` | committed |
| `tests/` | Metrics, model-call error reporting, SFT loss boundaries, rental shutdown, paired comparisons and the coding counterfactual | — |
| `external/` | Upstream Synthetic Hospital, cloned by `hospital/setup.sh` | gitignored |
| `runs/` | Full trajectories of every run | gitignored |

Write-ups: [milestone 1](results/milestone1.md) (benchmark audit and baselines),
[milestone 2](results/milestone2_other_tasks.md) (the other three tasks),
[milestone 3](results/milestone3_training.md) (SFT vs RL vs continued SFT; where the error is),
[a worked case](results/cases/patient_2123.md).

## Setup

```bash
hospital/setup.sh                 # clone upstream at the pinned commit, start the server on :58000
uv sync --group verifiers         # our code + the Verifiers environment
```

Put `OPENROUTER_API_KEY=...` in `.env` (gitignored). Every command runs from the repo root as
`uv run python -m <folder>.<module>`; each folder's README lists its commands.

On a GPU container without nested Docker, use `bash hospital/setup_native.sh`
and `bash training/setup_inference.sh` instead. This creates host-local secrets
and serves the exact BF16 checkpoint privately; no OpenRouter key is needed.
See [`training/README.md`](training/README.md) for evaluation and rental shutdown.

## Data splits

| Split | Patients | Use |
|---|--:|---|
| `train` | 680 | training (RL, SFT demonstrations, prompt optimization) |
| `dev` | 120 | checkpoint and model selection, audits |
| `public` | 200 | the paper's reported benchmark; comparison against published numbers; untouched final test of the SFT/RL/continued-SFT phase |
| `heldout` | 268 | sealed final evaluation; not reachable from our code |

`train` and `dev` partition the upstream 800-patient `train` split, stratified by dominant
ICD-10 chapter and encounter count (seed 20260930; `uv run python -m common.splits` rebuilds it).

## Metrics

`mean reward` is the mean per-episode `weighted_problem_list_f1_neutral`: what the environment
returns and what RL optimizes. `official F1` is the paper's batch aggregation (harmonic mean of
mean severity-weighted recall and mean chart-neutral precision), which silently excludes empty
submissions; `official F1 (strict)` counts them as zero. `recall, earlier / final visit` splits
recall by whether the diagnosis is written in a later note (copyable) or must be inferred.
See `common/metrics.py`.
