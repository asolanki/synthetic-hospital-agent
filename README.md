# Synthetic Hospital RL

This is the code behind my blog post, [A 4B model in a synthetic hospital: the agent is more than the weights](https://adarshsolanki.com/posts/synthetic-hospital/).

I took a small open model (Qwen3-4B-Instruct) and tried to train it on the patient diagnosis task from [Synthetic Hospital](https://github.com/sparkcpark/synthetic_hospital), a new benchmark of synthetic patients served through an EHR-style environment. The agent opens a patient's chart with tools and has to submit every diagnosis with its ICD-10 code.

The short version: RL barely helped (+0.01). What actually worked was fine-tuning plus a small ICD-10 search tool, which took the 4B from 0.17 to 0.74. GPT-6 Luna didn't need the tool, but two sentences in its prompt about what the answer key counts took it to 0.81.

The blog has the full story. This repo has the code, the data splits, the training examples, and every result.

## Results

F1 on the paper's 200 test patients (failed sessions count as zero).

| Setup | F1 |
|---|--:|
| Qwen3-4B, untrained | 0.17 |
| Qwen3-4B, SFT 3 epochs | 0.39 |
| Qwen3-4B, SFT 3 epochs + RL (GRPO) | 0.40 |
| Qwen3-4B, SFT 8 epochs | 0.53 |
| Qwen3-4B, untrained + search tool | 0.49 |
| **Qwen3-4B, tool-SFT + search tool** | **0.74** |
| Qwen3-30B-A3B, untrained | 0.46 |
| Qwen3-30B-A3B, untrained + search tool | 0.64 |
| Kimi K2.5 | 0.73 |
| GPT-6 Luna | 0.69 |
| GPT-6 Luna + search tool | 0.68 |
| **GPT-6 Luna + two sentences about the answer key** | **0.81** |
| Gemma 3 27B (lower bound, its API provider dropped a lot of replies) | 0.26 |
| Copy the chart's problem list and code it perfectly (a ceiling, not an agent) | 0.82 |

Every agent row uses the same clean harness (the benchmark's `/env` endpoints). The search tool and the prompt sentences are my additions, so treat comparisons with the paper's numbers as context, not a leaderboard.

## Trained adapter

The tool-SFT LoRA adapter (the bold 0.74 row, rank 16 on [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) at revision `cdbee75f`) is a 126 MB download: [qwen3-4b-sft-icd10-v1-adapter.tar.gz](https://files.adarshsolanki.com/sft-icd10-v1/qwen3-4b-sft-icd10-v1-adapter.tar.gz) (SHA-256 `40694f8be937b9ce5569af52ea3eb81267f02a271a5bb44510baeb0b23479059`). Extract it into `runs/` and serve it with the commands in [`training/README.md`](training/README.md#the-main-runs). It needs the search tool to reach that score.

## What's in here

| Folder | What it is |
|---|---|
| [`hospital/`](hospital/README.md) | Starts the benchmark server |
| [`audit/`](audit/README.md) | Poking at the benchmark before training: scorer checks, shortcuts, the harness leak, the label audit |
| [`baselines/`](baselines/README.md) | My first agent harness, kept so the earliest numbers reproduce |
| [`training/`](training/README.md) | The environment, the search tool, SFT, RL, and evaluation |
| `common/` | Small shared helpers: server client, database reader, splits, metrics |
| `data/splits/` | My train/validation split |
| `data/sft/` | The SFT training examples (with and without the search tool) |
| `results/` | Write-ups for each stage, and the raw JSON behind every number |
| `tests/` | Unit tests for the metrics, comparisons, and a few tricky bits |
| `LOG.md` | My lab log: every decision and prediction, timestamped, including the wrong ones |

## Getting started

You'll need Docker (for the benchmark server) and [uv](https://docs.astral.sh/uv/).

```bash
hospital/setup.sh                                 # clone the benchmark and start its server on :58000
uv sync --group verifiers                         # install this repo and the environment
uv run python -m training.icd10_codebook fetch    # download the ICD-10-CM code list and build the search index
```

Put API keys in a `.env` file (it's gitignored):

- `OPENROUTER_API_KEY` for the hosted models (Kimi, Luna, Qwen3-30B).
- `DEEPINFRA_API_KEY` only for Gemma 3 27B.

Everything runs from the repo root as `uv run python -m <folder>.<module>`. Each folder's README has its commands. Training needs a GPU. I used one rented A100 80GB, and [`training/README.md`](training/README.md) covers that setup.

## Data splits

| Split | Patients | What I used it for |
|---|--:|---|
| train | 680 | Training examples and RL |
| validation (`dev` in the code) | 120 | Every decision: picking checkpoints, comparing setups |
| test (`public` in the code) | 200 | The paper's public patients. Run once per model at the end |
| held-out | 268 | The paper's private test set. Never touched |

Train and validation are my split of the paper's 800 training patients (`uv run python -m common.splits` rebuilds it).

## Write-ups

The `results/` folder has a write-up for each stage, with the confidence intervals and details the blog skips:

1. [Auditing the benchmark](results/milestone1.md)
2. [The other three tasks](results/milestone2_other_tasks.md)
3. [SFT vs RL](results/milestone3_training.md)
4. [Training longer, and the search tool](results/milestone4_sft_and_codebook.md)
5. [Against frontier models](results/milestone5_head_to_head.md)

Heads up: these were written as I went, so they use some older names. "Code book" means the search tool, "house style" means the answer key's conventions, "dev" is validation, and "public" is test.

## How I built this

I chose the questions and made the calls, and coding agents did most of the engineering. Everything is in [`LOG.md`](LOG.md).
