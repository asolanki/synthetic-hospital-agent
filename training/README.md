# training/

Everything for running, fine-tuning, and RL-training the agent. I ran all of it on one rented A100 80GB (Vast.ai, about $1 an hour). The nitty-gritty (exact flags, how the scripts guard against mistakes, W&B setup) is in [DETAILS.md](DETAILS.md). This page is the short version.

## The pieces

| Path | What it is |
|---|---|
| `environment/` | The benchmark wrapped as a [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment. The same environment is used for evaluation, SFT data, and RL. It's its own package because the RL trainer loads it by name |
| `environment/.../icd10.py`, `icd10_codebook.py` | The ICD-10 search tool: plain keyword (BM25) search over the official CMS 2025 code list |
| `sft_data.py`, `sft_data_icd10.py` | Build the SFT training examples (the three-step routine, and the same routine with search steps) |
| `sft.py`, `run_sft.sh`, `setup_sft.sh` | LoRA fine-tuning |
| `run_rl.sh`, `setup_rl.sh`, `serve_rl.sh` | GRPO with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), plus a small patch for a shutdown hang (`prime-rl-shutdown.patch`) |
| `serve.sh`, `setup_inference.sh` | Serve a model (or a LoRA adapter) with vLLM |
| `evaluate.py`, `report.py` | Run an evaluation and turn the transcripts into a scored report |
| `compare.py`, `breakdown.py`, `interaction.py` | Compare two runs patient by patient, with bootstrap confidence intervals, and break down where they differ |
| `sampling_signal.py` | The "does RL have anything to learn?" test: how often 4 attempts at the same patient disagree |
| `lookups.py`, `recode.py` | How the model used the search tool, and what its score would be with perfect codes |
| `gemma_toolcall_proxy.py` | A tiny proxy that turns Gemma 3's text tool calls into real ones (otherwise its calls never reach the harness) |
| `configs/` | Every evaluation, SFT, and RL config I ran |

## Setup

```bash
uv sync --group verifiers
set -a; source .env; set +a
```

`vf-eval` needs `uv` 0.9 or newer in `~/.local/bin` (`uv self update`). On a GPU box without Docker, start the server with `bash hospital/setup_native.sh` (see [`hospital/`](../hospital/README.md)) and the model server with `bash training/setup_inference.sh`.

## The main runs

**Fine-tuning (3 epochs, the first SFT run):**

```bash
bash training/setup_sft.sh
bash training/run_sft.sh
```

**RL (GRPO from the 3-epoch checkpoint):**

```bash
bash training/setup_rl.sh
bash training/run_rl.sh
```

**Training longer (10 epochs, keeping every checkpoint. I picked epoch 8 on validation):**

```bash
bash training/run_sft.sh --config training/configs/sft_sweep.toml --no-merge
```

**The search tool and tool-SFT:**

```bash
uv run python -m training.icd10_codebook fetch     # download the code list, build the index
uv run python -m training.sft_data_icd10 --source oracle-copy-v1 --name oracle-copy-icd10-v1 \
    --tools-from data/sft/oracle-copy-icd10-v1/tools_source_trace.jsonl
bash training/run_sft.sh --config training/configs/sft_icd10.toml --no-merge
```

**Evaluating a model:**

```bash
bash training/serve.sh Qwen/Qwen3-4B-Instruct-2507 shrl-sft-icd10-v1 "$PWD/runs/sft-icd10-v1/adapter"
uv run --group verifiers --group tracking python -m training.evaluate \
    --config training/configs/eval_local_icd10.toml --model shrl-sft-icd10-v1 \
    --split public --name qwen3-4b-sft-icd10-public --output-dir runs/icd10-public/tool-sft
```

`--split dev` is validation and `--split public` is test. Small adapters are served unmerged on the base model (merging a LoRA into bf16 weights can erase it, see the blog's side quests).

**The frontier models (hosted, no GPU needed):**

```bash
uv run vf-eval @ training/configs/eval_public_kimi_k25.toml -n 200 -c 24 --output-dir runs/eval-public-kimi
uv run vf-eval @ training/configs/eval_public_gpt6_luna.toml -n 200 -c 16 --output-dir runs/eval-public-luna
uv run vf-eval @ training/configs/eval_public_gpt6_luna_style.toml -n 200 -c 16 --output-dir runs/eval-public-luna-style
```

The `_style` config is Luna with the two extra sentences about what the answer key counts.

**Comparing two runs:**

```bash
uv run python -m training.compare --split public \
    --reference results/raw/training/gpt-6-luna-env-public.json \
    --candidate results/raw/training/qwen3-4b-sft-icd10-public.json \
    --name qwen3-4b-sft-icd10-vs-gpt-6-luna-env-public
```

The saved reports for every run are already in `results/raw/training/`, so you can rerun any comparison without a GPU.
