# Synthetic Hospital RL

What does RL add over strong prompting and supervised fine-tuning when a small open-weight
model has to operate a longitudinal EHR environment? See [PLAN.md](PLAN.md) for the project
plan and [results/milestone1.md](results/milestone1.md) for the current findings.

Task: `patient_diagnosis` in [Synthetic Hospital](https://github.com/sparkcpark/synthetic_hospital)
v1.3, played through the corrected server-side-filtered `/env` route (never the original
`eval/agents` harness, which leaks the labels through the problem list).

## Setup

```bash
scripts/setup_env.sh      # clone Synthetic Hospital at the pinned commit, start the simulator on :58000
uv sync
```

Put `OPENROUTER_API_KEY=...` in `.env` (gitignored) to run models through OpenRouter, or point
`--base-url` at any OpenAI-compatible server (vLLM, LM Studio).

## Commands

| Command | What it does |
|---|---|
| `uv run shrl verify` | Phase 1: oracle scores 1.0 everywhere, malformed submissions, budget enforcement, label exposure in observations |
| `uv run shrl split` | Writes the fixed train/dev split to `splits/patient_diagnosis.json` |
| `uv run shrl audit --split dev` | Phase 2: scores hand-written strategies (extraction, priors, oracle perturbations) |
| `uv run shrl baseline --model <id> --split dev` | Rolls a chat model out in the agent environment; full trajectories in `runs/<name>/` |
| `uv run shrl report` | Rebuilds `results/tables.md` from `results/*.json` |

## Data splits

| Split | Patients | Use |
|---|--:|---|
| `train` | 680 | training (RL, SFT trajectories, prompt optimization) |
| `dev` | 120 | checkpoint and model selection, audit |
| `public` | 200 | the paper's reported benchmark; comparison against published numbers |
| `heldout` | 268 | sealed final evaluation; not reachable from the CLI |

`train` and `dev` partition the upstream 800-patient `train` split, stratified by dominant
ICD-10 chapter and encounter count (seed 20260930).

## Metrics

`mean reward` is the mean per-episode `weighted_problem_list_f1_neutral`, which is what the
environment returns and what RL optimizes. `official F1` is the paper's batch aggregation
(harmonic mean of mean severity-weighted recall and mean chart-neutral precision), which
silently excludes empty submissions; `official F1 (strict)` counts them as zero. See
`shrl/metrics.py`.
