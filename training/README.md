# training/ — Phases 3b–4: model evaluation and training

Evaluation and RL share the Verifiers environment and official score.
SFT uses recorded demonstrations from that environment with Transformers/PEFT;
it does not query the scorer during gradient updates.

| Path | What it is |
|---|---|
| `environment/` | The benchmark as a [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 package (`synthetic-hospital-dx`), which prime-rl loads by name. Standalone on purpose: it does not import `common/`. |
| `sft_data.py` | Builds SFT demonstrations into `data/sft/<name>/` |
| `sft.py`, `configs/sft_oracle_copy.toml` | Train-only, assistant-masked LoRA SFT; adapter and merged checkpoint export |
| `configs/sft_continued.toml` | Continued-SFT control: the same recipe as a fresh adapter on the merged SFT checkpoint |
| `setup_sft.sh`, `run_sft.sh`, `sft-requirements.txt` | Isolated, pinned CUDA 12.8 training environment |
| `configs/` | OpenRouter/local evaluation, SFT and RL configurations |
| `setup_inference.sh`, `serve.sh` | Pinned CUDA 12.8 vLLM environment and private Qwen server |
| `inference-requirements.txt` | Exact resolved inference dependencies, separate from the root `uv.lock` |
| `throughput.py` | Real dev-rollout concurrency sweep, resource samples and per-trial traces |
| `report.py` | Recover saved submissions, verify their official scores, and build evaluation summaries |
| `evaluate.py`, `tracking.py` | Full-dev (or train-fit) evaluation with hosted W&B curves, patient tables, transcript media and raw artifacts |
| `compare.py`, `breakdown.py` | Patient-paired checkpoint comparisons and where-did-it-change slices |
| `interaction.py` | Patient-paired difference in differences of four reports (e.g. is the code book worth more to the 4B than to a frontier model?), on the same bootstrap as `compare.py` |
| `lookups.py` | Code-book use in saved traces: lookups per patient, repeated queries, and an error budget (hit and shown, hit from memory, missed though shown, never shown) |
| `recode.py` | Coding counterfactual: rescore a report with the benchmark's codes for the names the model chose |
| `setup_rl.sh`, `serve_rl.sh`, `run_rl.sh` | Isolated Prime-RL/CUDA 13 stack, colocated inference and training |
| `prime-rl-shutdown.patch` | Pinned fix for the final-broadcast queue-drain deadlock |
| `track_rl.py` | Publish native RL episode chunks into the same hosted W&B run |
| `rental_guard.py` | Stop the current Vast instance at an absolute deadline using its instance-scoped key |
| `icd10_codebook.py` | `fetch`: download and checksum the public CMS ICD-10-CM 2025 release and build the search index; `check`: name-search recall on dev and a top-result recode of saved reports |
| `sft_data_icd10.py` | Tool-SFT demonstrations: one inserted `icd10_search` turn per oracle-copy demonstration, submission unchanged |
| `sampling_signal.py` | RL-signal test: how often k sampled answers per patient agree (would GRPO have anything to learn?) |
| `gemma_toolcall_proxy.py` | Local OpenAI-compatible proxy in front of DeepInfra that turns Gemma 3's text-format tool calls into structured ones |
| `configs/sft_sweep.toml` | Epoch sweep: the unchanged recipe for 10 epochs, every epoch's adapter kept |
| `configs/sft_icd10.toml` | Tool-SFT: the unchanged recipe, 8 epochs, on `oracle-copy-icd10-v1` |
| `configs/eval_local_icd10.toml` | `eval_local.toml` plus the code-book tool |
| `configs/eval_openrouter_icd10_t0.toml` | Qwen3-30B with the code-book tool at temperature 0; pairs with `audit/harness_gap_env.toml`, which is the same run without it |
| `configs/eval_public_kimi_k25.toml`, `eval_public_gpt6_luna.toml`, `eval_public_gemma3_27b.toml` | No-tool comparators on the 200 public patients |
| `configs/eval_public_gpt6_luna_icd10.toml` | GPT-6 Luna plus only the code-book tool; completes the 2×2 {4B SFT, 4B tool-SFT} × {Luna, Luna + book} |
| `configs/rl_inference.toml` | Prime-RL rollout inference server (RL only; checkpoint evaluation stays on vLLM 0.19.1) |

## The environment package

| File | Role in an episode |
|---|---|
| `taskset.py` | One task per patient. `setup` opens a simulator episode; `finalize` closes it; the reward is the official per-episode score |
| `tools.py` | The paper's 10 EHR tools, served over MCP; each call is forwarded to `/env/step` |
| `env.py` | The paper's control flow: nudge after a reply with no tool call, then one forced-submission turn |
| `state.py` | Per-episode bookkeeping (episode id, actions, submitted, reward) |
| `simulator.py` | Small HTTP helper (its own copy of the server address and token) |
| `icd10.py` | BM25 search over the public CMS ICD-10-CM 2025 release: tabular titles, inclusion terms, alphabetic index, Neoplasm Table, Table of Drugs and Chemicals. Never the benchmark's diagnosis table |
| `icd10_tool.py` | The optional `icd10_search` tool. Off by default; `[env.taskset.task] icd10_tool = true` enables it. Lookups run locally and spend model turns, not EHR actions |

## Commands

```bash
uv sync --group verifiers
set -a; source .env; set +a
uv run vf-eval @ training/configs/eval_openrouter.toml -n 5 --output-dir runs/vf-smoke
uv run python -m training.sft_data --name oracle-copy-v1 \
    --tools-from data/sft/oracle-copy-v1/tools_source_trace.jsonl
```

`vf-eval` runs the null harness, which calls `uv sync --script` with `~/.local/bin` first on
`PATH`; the `uv` there must be 0.9 or newer (`uv self update`).

## Validation

On the 120 dev patients, Qwen3-30B scores 0.466 here and 0.463 in `baselines/agent.py`
(paired difference +0.003, 95% CI −0.04 to +0.05). Without `env.py` it scored 0.263,
because 47 episodes ended when the model wrote its answer as text instead of calling the tool.

## Self-hosted 4B evaluation

After [native hospital setup](../hospital/README.md#gpu-containers-without-docker):

```bash
bash training/setup_inference.sh
set -a; source .env; set +a
uv run --locked vf-eval @ training/configs/eval_local.toml -n 3 -c 3 \
    --no-rich --output-dir runs/local-smoke --run.dir smoke
uv run --locked python -m training.report runs/local-smoke/smoke/traces.jsonl \
    --name local-smoke
uv run --locked python -m training.throughput \
    --tasks 64 --concurrency 8 16 32 64 --output-dir runs/throughput
```

Full dev evaluation (120 patients, unchanged paper prompt; no training):

```bash
uv run --locked vf-eval @ training/configs/eval_local.toml -n 120 -c 32 \
    --no-rich --output-dir runs/local-paper --run.dir eval
uv run --locked python -m training.report runs/local-paper/eval/traces.jsonl \
    --name qwen3-4b-paper-dev
uv run --locked python -m baselines.report
```

The recorded A100 runs use `throughput.run_trial` around these evaluations,
adding isolated cache salts, exact wall times, resource samples, and saved
JSON configs under `runs/a100-dev-v019/{paper,strong_v1}/`. The corresponding
runtime manifest is `results/raw/training/a100-runtime.json`. Concurrency 32
is the conservative full-evaluation setting, not a demonstrated optimum.

The server binds to `127.0.0.1:8000`; use SSH forwarding rather than publishing an
unauthenticated inference port. `supervisorctl status shrl-vllm` shows its state;
logs are in `runs/services/vllm.log`. This is standalone inference, not a
prime-rl trainer installation.

The checkpoint is `Qwen/Qwen3-4B-Instruct-2507` at Hugging Face revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, loaded in BF16 without quantization.
vLLM 0.19.1 / PyTorch 2.10.0 use CUDA 12.8, verified on this host's driver.
Transformers is pinned to 4.57.1 in the inference environment; the root Verifiers
environment retains its existing lockfile. Serving uses the model's own chat
template, Hermes tool parsing, prefix caching, a 65,536-token context limit,
64 scheduler slots, and 90% GPU memory utilization.

Do not use the initial vLLM 0.11.0 deployment for headline results: its Hermes
stream parser truncated tool arguments when output chunks coalesced under load
([upstream defect](https://github.com/vllm-project/vllm/issues/32152)). The
0.19.1 release replaces that parser. The original sweep and raw wire captures
remain in `runs/a100-throughput/` and `runs/a100-protocol-probe/`; they are
serving diagnostics, not a clean model baseline.

Sampling matches the existing eval's temperature 0.7, top-p 0.8, and 2,048-token
per-turn limit, with seed 0. `--generation-config vllm` prevents hidden
checkpoint generation defaults from changing those controls. No extra top-k,
repetition penalty, forced tool choice, or reward modification is applied.

The sweep gives each trial a new cache salt while retaining prefix caching
within the trial. Do not run other inference clients concurrently with it.
Wall time includes harness startup and the last slow episode; GPU utilization
and KV occupancy are sampled every two seconds. The hospital latency sample is
the health endpoint, not tool-call latency.
Provider/harness failures remain in the reward denominator with zero reward;
they are not retried to improve the result. A completed/submitted episode is
not necessarily scorer-valid or correct. Missing finalization metrics are
reported separately: action/nudge averages and budget-exhaustion rates use
episodes with observed metrics, not fabricated zeros.
`rollout_error_rate` follows the harness's episode status. Also inspect
`model_call_error_episode_rate` and `model_call_error_types`: handled model-call
errors, including context-limit termination, can occur in an episode marked
`ok`. The recorded base comparisons use a 65,536-token serving limit.

`report.py` saves `results/raw/training/<name>.json`. It verifies that recovered
answers rescore identically, counts failures in the strict aggregate, and
reports both provider token totals (history counted on every request) and
Verifiers' unique-history token totals. Only reports covering an entire fixed
split enter `uv run python -m baselines.report`; smoke/sweep subsets do not.
Strong-prompt experiments are complete and retired from the active workflow.
Keep their saved results as historical evidence; use only the paper prompt for
SFT and RL checkpoint comparisons.

## SFT control

The initial control uses all 680 existing `oracle-copy-v1` demonstrations,
with the paper prompt and the exact evaluation tool schemas. These are
label-informed demonstrations: their final diagnosis is the reference answer,
including final-visit labels not explicitly named in the note. They are not
samples of an independently successful model policy.

`sft.py` rejects duplicate, missing, dev or unknown patient ids and rejects
overlong examples rather than truncating the final submission. It uses the
checkpoint's native chat template and verifies prefix consistency for each
assistant turn. Loss covers generated assistant tokens and their end-of-turn
token, excluding deterministic role headers, system/user/tool context and
post-EOS template newlines. The final scorer response is omitted entirely.

```bash
bash training/setup_sft.sh
HF_HOME=/workspace/.hf_home .venv-sft/bin/python -m pytest -q tests/test_sft.py
bash training/run_sft.sh --prepare-only --output-dir runs/sft-preparation

# One GPU: do not run inference and training concurrently.
supervisorctl stop shrl-vllm
bash training/run_sft.sh --max-steps 2 --no-merge --output-dir runs/sft-setup-check
bash training/run_sft.sh
```

Every invocation requires a new output directory. The full run reloads the
original base checkpoint; it does not continue from the setup-check adapter.
The fixed control is three epochs, LoRA rank 16 / alpha 32 / dropout 0.05,
learning rate 1e-4, effective batch 8 and seed 0. It is configured for one A100
with BF16, SDPA and gradient checkpointing.

Each run saves its configuration, source/data/split/template hashes, package
versions and token counts in `manifest.json`; loss/gradient logs and optimizer
state under `checkpoints/`; final PEFT weights under `adapter/`; and a merged
BF16 checkpoint under `merged/` unless `--no-merge` is set. `result.json` records
training metrics and peak Torch GPU memory. No dev examples enter training.

Training streams loss, gradient norm, learning rate and GPU metrics to the
private W&B project configured in `configs/sft_oracle_copy.toml`.
Authenticate with `uvx --from wandb==0.26.1 wandb login --verify`; enter credentials
only in the CLI prompt, never in source code or command arguments.
Each run records its W&B URL in the manifest and result. The final adapter,
configuration, implementation and loss history are uploaded as one model artifact;
the full merged checkpoint stays on disk. Use `--wandb-mode offline` or
`--wandb-mode disabled` explicitly for a disconnected setup check.

Serve the merged checkpoint with the same inference settings:

```bash
bash training/serve.sh "$PWD/runs/sft-oracle-copy-v1/merged" \
    shrl-qwen3-4b-sft-oracle-copy-v1
```

This command runs in the foreground; under Supervisor, set `shrl-vllm`'s
command to the same invocation rather than starting a second server.
Evaluate with the full-dev commands above, adding
`--model shrl-qwen3-4b-sft-oracle-copy-v1` and fresh output directories.
Keep the paper prompt fixed. The root Verifiers lock and the inference environment
remain separate from trainer dependencies.

## RL from the preserved SFT control

RL freezes `runs/sft-oracle-copy-v1/merged` and trains a **new** rank-16,
alpha-32 LoRA adapter. It does not overwrite the SFT checkpoint or continue the
setup-check adapter. `configs/rl_diagnosis.toml` declares 85 updates, 32 rollouts
per update, four rollouts per patient, learning rate 1e-5 and dropout 0.
The native GRPO variant centers rewards within each group without dividing by
their standard deviation, then uses a PPO-clipped objective (ratio 0.8–1.2).
There is no reference-model KL term, length penalty or reward shaping.

Use only the 680 training patients and the unchanged published paper prompt.
Training samples at temperature 1 / top-p 1; checkpoint evaluation retains the
original temperature 0.7 / top-p 0.8 controls. Dev is for comparison, not gradient
updates; heldout remains sealed. One SFT recipe and one RL pilot do not establish
either method's ceiling. Assess continued SFT and bounded tuning before making a
method-level claim about RL.

The isolated stack is Prime-RL revision
`d9adc1e867db9f8e27438fdd65f99e8dc7a2448a` **plus**
`prime-rl-shutdown.patch`, with Torch 2.13/CUDA 13, vLLM 0.30 and FlashAttention 2.
CUDA forward compatibility is process-local; the system driver and CUDA 12.8
evaluation environment are unchanged. The patch keeps the final weight barrier
after queue draining, rather than blocking the queue consumer before the watcher
can enqueue cancellations. The two-update run exercised the failure before the
patch and completed with a terminal checkpoint afterward.

On the configured A100 rental:

```bash
bash training/setup_rl.sh
supervisorctl stop shrl-vllm
supervisorctl restart shrl-rl-inference
bash training/run_rl.sh --max-steps 2 --run.name rl-setup-check --ckpt.interval 2

# A fresh independent run: clear the version-salted KV cache by restarting inference.
supervisorctl restart shrl-rl-inference
bash training/run_rl.sh
```

The Supervisor service runs `bash training/serve_rl.sh`; the dependency setup
helper does not install that service. Wait for its private port-8000 health
endpoint before starting training. Inference and training share GPU 0; inference
reserves 40% of GPU memory and uses eight scheduler slots. The native router is
disabled because there is only one engine. Do not add a launcher `[inference]`
block: that deployment mode allocates disjoint GPUs.

Native run directories are `runs/rl/<run.name>/`. Checkpoints include the frozen
base and optimizer state and currently occupy about 17 GiB each; keeping one
still requires room for two while the next checkpoint is written. Preserve
verified local backups before pruning redundant rental copies.

Evaluate the **unmerged** RL adapter on the preserved SFT base with the original
vLLM 0.19.1 evaluation environment:

```bash
bash training/serve.sh "$PWD/runs/sft-oracle-copy-v1/merged" \
    shrl-qwen3-4b-rl-sft-paper-v1 \
    "$PWD/runs/rl/rl-sft-paper-v1/broadcasts/step_85"
```

Stop native RL inference first; do not run both servers on the A100. The adapter
and base have distinct API names so an adapter request cannot resolve to the
unmodified base. Do not merge these small updates into BF16 weights: the
two-update check measured 98.6–99.3% relative L2 error in the intended weight
delta across the first layer's seven projections after rounding
(`rl-merge-precision-check.json`). Runtime LoRA avoids that base-weight rounding;
its inference matrices still use BF16, as in native RL inference.

`results/raw/training/rl-setup-check.json` records actual reward rescoring,
sampled-token masks, token-usage alignment, finite log probabilities and changed
adapter tensors. Failed and stale rollouts remain in the native artifact stream
but may be excluded from gradient updates; report actual coverage and exclusions
rather than calling 85 updates an exact pass over every patient.

Native W&B curves are live. To add episode tables, run `training.track_rl` with
the Prime-RL Python environment and the native run directory. Its `--pid` option
follows a local launcher until exit; omit it to backfill a completed run. It
handles compressed chunks and preserves an available startup adapter before
native pruning. During live training, the native process owns run finalization.
Backfill first records the terminal native outcome and restores that same outcome
on exit, because opening a shared W&B writer can reopen a finished run. A completed
backfill is not published twice; interrupted partial backfills are not resumable.
Tables include **all finalized episodes**, not only those admitted into updates. Their mean
reward is therefore distinct from the native effective-batch training curve.

Compare complete dev reports without regenerating answers:

```bash
uv run python -m training.compare \
    --reference results/raw/training/qwen3-4b-sft-paper-dev.json \
    --candidate results/raw/training/qwen3-4b-rl-paper-dev.json \
    --name qwen3-4b-rl-vs-sft-paper
```

The comparison pairs all 120 fixed dev patients, retains failures, and saves
patient deltas plus a 1,000-resample, seed-0 bootstrap interval. It rejects
incomplete, duplicated or non-dev reports. These are dev comparisons, not an
unbiased estimate after repeated recipe selection.
The interval resamples patients, not training seeds. Pass several reports per
arm (e.g. sampling seeds 0–4) to average each patient's reward within an arm
before pairing; the same report cannot count twice.

`training.breakdown` takes the same arguments and shows where a change came from:
unweighted diagnosis recall by visit (earlier encounters vs the final one) and by
how many training patients carry the same match key (unseen = none), each reference
diagnosis's outcome (hit, named but miscoded, or not named), paired reward by
encounter-count bucket, behaviour summaries, and answer changes. Both accept
`--split public` for the paper's reported split.

`training.recode REPORT --name NAME` rescores a saved report after replacing each
submitted entry's code with the benchmark table's code for that exact name. It is a
diagnostic counterfactual that separates ICD coding from chart reading, not an agent score.

For native training coverage and curves, use the Prime-RL environment:

```bash
external/prime-rl/.venv/bin/python -m training.rl_report \
    runs/rl/rl-sft-paper-v1 --expected-steps 85 --name rl-sft-paper-v1-training
```

This joins arrival records with actual ship annotations, checks all 32-rollout
batches and trainer updates, and rejects stale/mixed-policy shipped traces.
All-finalized and shipped populations are reported separately. Native windowed
cancellation/error/admission metrics are retained, not forced into a disjoint
partition of all saved episodes. On the completed setup run, the reconstructed
64 shipped episodes' per-update reward means match the native effective curves.

## Epoch sweep

One fresh run of the unchanged oracle-copy recipe from the base model for 10 epochs on one cosine
schedule, keeping every epoch's adapter (`keep_epoch_checkpoints = true`; default off). Each epoch is
scored on dev at sampling seeds 0–2; the rule (highest three-seed mean, earliest epoch within 0.005)
was logged beforehand and picked epoch 8 (`results/raw/training/sft-sweep-selection.json`).

```bash
bash training/run_sft.sh --config training/configs/sft_sweep.toml --no-merge
# 680 demonstrations / effective batch 8 = 85 optimizer steps per epoch; epoch 8 = checkpoint-680.
bash training/serve.sh Qwen/Qwen3-4B-Instruct-2507 shrl-sft-sweep-e8 \
    "$PWD/runs/sft-sweep-v1/checkpoints/checkpoint-680"
for s in 0 1 2; do
  uv run --group verifiers --group tracking python -m training.evaluate \
      --model shrl-sft-sweep-e8 --seed $s --mode offline \
      --name qwen3-4b-sft-sweep-e8-paper-dev-seed$s --output-dir runs/sft-sweep-dev/e8-seed$s
done
```

Adapters are served unmerged on the base checkpoint. Pass all three seed reports per arm to
`training.compare` to average each patient before pairing. Results: `results/milestone4_sft_and_codebook.md`.

## ICD-10-CM code-book tool and tool-SFT

`icd10_search(query)` searches the public CMS ICD-10-CM 2025 release only (zip SHA-256 pinned in
`icd10.py`). The benchmark's own diagnosis table is never indexed, so the tool cannot reveal a
patient's reference codes. The frozen index is `runs/icd10cm/search-index-2025-v3.json`
(gitignored; `SHRL_ICD10_DIR` overrides the directory). It is an augmented harness, off by default,
so official-harness results are unchanged.

```bash
uv run python -m training.icd10_codebook fetch          # download + verify the zip, build the index
uv run python -m training.icd10_codebook check --name icd10-codebook-v3-check-dev \
    results/raw/training/qwen3-4b-{paper,sft-paper,rl-paper,sft-continued-paper}-dev.json

# Tool-SFT demonstrations, training, serving and evaluation.
uv run python -m training.sft_data_icd10 --source oracle-copy-v1 --name oracle-copy-icd10-v1 \
    --tools-from data/sft/oracle-copy-icd10-v1/tools_source_trace.jsonl
bash training/run_sft.sh --config training/configs/sft_icd10.toml --no-merge
bash training/serve.sh Qwen/Qwen3-4B-Instruct-2507 shrl-sft-icd10-v1 "$PWD/runs/sft-icd10-v1/adapter"
uv run --group verifiers --group tracking python -m training.evaluate \
    --config training/configs/eval_local_icd10.toml --model shrl-sft-icd10-v1 \
    --split public --name qwen3-4b-sft-icd10-public --output-dir runs/icd10-public/tool-sft

# Untrained Qwen3-30B with the tool (OpenRouter), 120 dev patients.
uv run vf-eval @ training/configs/eval_openrouter_icd10_t0.toml -n 120 -c 8 --no-rich \
    --output-dir runs/icd10-dev-qwen30b-t0 --run.dir eval
uv run python -m training.report runs/icd10-dev-qwen30b-t0/eval/traces.jsonl --name qwen3-30b-icd10-dev-t0
```

Each tool-SFT demonstration keeps its recorded chart observations and reference submission and
inserts one turn that searches every submitted name, followed by the tool's real output. When a
search misses the reference category, the demonstration still submits the reference code
(label-informed, like the source data). Train patients only.

## RL-signal sampling test

Would GRPO from a policy have anything to learn? Draw k = 4 samples per patient at the RL rollout
controls (temperature 1, top-p 1, seeds 0–3), then count, per reference diagnosis, groups that are
0/4, mixed or 4/4, plus equal-reward patient groups. Reads saved reports only.

```bash
for s in 0 1 2 3; do
  uv run --group verifiers --group tracking python -m training.evaluate \
      --config training/configs/eval_local_icd10.toml --model shrl-sft-icd10-v1 \
      --split train --temperature 1 --top-p 1 --seed $s --mode offline \
      --name qwen3-4b-sft-icd10-train-t1-seed$s --output-dir runs/rl-signal/qwen3-4b-sft-icd10-train-seed$s
done
uv run python -m training.sampling_signal --split train --name qwen3-4b-sft-icd10-rl-signal-train \
    results/raw/training/qwen3-4b-sft-icd10-train-t1-seed{0,1,2,3}.json
```

Use the default config (no tool) for the no-tool SFT policies. The rule and result are in `LOG.md`
(2026-10-06 and 2026-10-07): on its own training patients tool-SFT has almost no signal.

## No-tool comparators (head-to-head)

Large models as agents in the same clean `/env` harness, without the code-book tool, on all 200
public patients, one sample each. Settings are in each config's header; the question and decision
rule were logged first (`LOG.md`, 2026-10-07). Kimi and Luna use `OPENROUTER_API_KEY`; Gemma uses
`DEEPINFRA_API_KEY` through the tool-call translator, which must be running first.

```bash
set -a; source .env; set +a
uv run vf-eval @ training/configs/eval_public_kimi_k25.toml -n 200 -c 24 --output-dir runs/eval-public-kimi
uv run vf-eval @ training/configs/eval_public_gpt6_luna.toml -n 200 -c 16 --output-dir runs/eval-public-luna

uv run python training/gemma_toolcall_proxy.py --port 58100 --log runs/gemma-proxy.jsonl   # separate shell
uv run vf-eval @ training/configs/eval_public_gemma3_27b.toml -n 200 -c 8 --output-dir runs/eval-public-gemma

# vf-eval writes runs/<dir>/<run-id>/traces.jsonl.
uv run python -m training.report runs/eval-public-kimi/<run-id>/traces.jsonl \
    --split public --name kimi-k2.5-env-public
uv run python -m training.compare --split public \
    --reference results/raw/training/kimi-k2.5-env-public.json \
    --candidate results/raw/training/qwen3-4b-sft-icd10-public.json \
    --name qwen3-4b-sft-icd10-vs-kimi-k2.5-env-public
uv run python -m training.breakdown --split public \
    --reference results/raw/training/kimi-k2.5-env-public.json \
    --candidate results/raw/training/qwen3-4b-sft-icd10-public.json \
    --name qwen3-4b-sft-icd10-vs-kimi-k2.5-env-public-breakdown
```

The proxy only rewrites responses; requests pass through unchanged, and every message it inspects
is appended to its JSONL audit log. `training/configs/eval_public_gpt6_luna_icd10.toml` is the same
Luna arm plus only the code-book tool. Results: `results/milestone5_head_to_head.md`.

## Private checkpoint archives

Useful checkpoints and reproduction bundles are stored in the private R2 bucket
`synthetic-hospital-rl-artifacts`, using the locally authenticated `cf` CLI.
The bucket has no enabled `r2.dev` or custom domain. Never copy Cloudflare
credentials to the rental or include `.env`, `.netrc` or credential caches.

Prepare a verified tar/gzip stream from explicit checkpoint, code, configuration
and provenance paths; split it into **16 MiB** `archive.tar.gz.part-*` files.
Use `tar -cf - | gzip -c` rather than BSD tar's compressed stdout, which pads the
stream with zeros. The CLI has a 30-second request deadline, so the REST endpoint's
300 MB size limit is not a practical upload chunk size on this connection.

```bash
python3 -m training.archive_r2 runs/archives/NAME \
    --bucket synthetic-hospital-rl-artifacts --prefix DATE/NAME
```

Re-running reuses only parts whose remote size and ETag match local bytes/MD5;
conflicting parts fail. After every part passes a fresh listing check, the
uploader publishes `manifest.json` with ordered keys, per-part SHA-256 hashes
and the combined archive hash. A failed upload is not a completed archive.

For restoration, `cf r2 objects get KEY --bucket-name BUCKET` writes raw bytes
to stdout; save the manifest and each listed part to files. Verify part SHA-256
hashes, concatenate in manifest order, verify the combined SHA-256, then extract
into an empty directory. Preserve the source-file checksums too.

The completed SFT archive is under
`2026-10-04/sft-oracle-copy-v1/chunks-16m/`: **412 parts, 6,904,606,720 bytes**.
Its durable restoration manifest and upload/download evidence are
`results/raw/training/sft-r2-manifest.json` and `sft-r2-archive.json`.
All parts passed size/ETag checks; downloaded manifest/first/last parts were
verified separately. Local restoration reproduced all 41 source-file hashes.

Archives from 2026-10-05 onward (same format; per-file source hashes in
`results/raw/training/NAME-files.json`, evidence in `NAME-r2-manifest.json` and
`NAME-r2-archive.json`). Each tar member was decompressed and checked against
hashes taken on the rental before upload.

| Prefix | NAME | Contents | Parts |
|---|---|---|--:|
| `2026-10-05/rl-sft-paper-v1/essentials-16m/` | `rl-sft-paper-v1-essentials` | RL adapters (updates 0, 84, 85), configs, logs, native traces/metrics, RL code and patch | 20 |
| `2026-10-05/rl-sft-paper-v1/checkpoint-step85-16m/` | `rl-sft-paper-v1-checkpoint` | Terminal trainer + optimizer checkpoint, enough to resume RL | 515 |
| `2026-10-05/sft-continued-v1/chunks-16m/` | `sft-continued-v1` | Continued-SFT adapter, trainer checkpoint, manifest, config and code | 30 |
| `2026-10-05/post-rl-evaluations/chunks-16m/` | `post-rl-evaluations` | Every dev/train/public evaluation trace from the RL phase, reports, service logs and chain scripts | 2 |
| `2026-10-06/sft-sweep-v1/chunks-16m/` | `sft-sweep-v1` | All 10 epoch adapters and the final adapter of the sweep | 83 |
| `2026-10-06/sft-icd10-v1/chunks-16m/` | `sft-icd10-v1` | Tool-SFT adapter, tool demonstrations, code-book zip and search code | 17 |
| `2026-10-06/sweep-icd10-evaluations/chunks-16m/` | `sweep-icd10-evaluations` | Every sweep and tool-arm evaluation directory, rental `results/raw`, chain scripts and logs | 4 |
| `2026-10-06/rl-signal-sampling/chunks-16m/` | `rl-signal-sampling` | RL-signal sampling runs, chain script and log, evaluator and analysis code | 4 |


## Watching experiments

The private hosted project is
[`grid-labs-llc/synthetic-hospital-rl`](https://wandb.ai/grid-labs-llc/synthetic-hospital-rl).
It remains available after the rental stops; no SSH tunnel is needed.
The workspace's **train** section shows SFT loss, gradient norm and learning rate.
**Tables / rollouts** contains completed patients, including failures, with
expandable transcript media. The same tables are available under each run's
**Artifacts** tab. Raw JSONL traces and the canonical report are evaluation artifacts.
SFT replays recorded demonstrations; it does not produce new hospital rollouts.

With a checkpoint served privately, evaluate the unchanged paper prompt:

```bash
uv sync --group verifiers --group tracking
set -a; source .env; set +a
uv run --group verifiers --group tracking python -m training.evaluate \
    --model shrl-qwen3-4b-sft-oracle-copy-v1 \
    --name qwen3-4b-sft-paper-dev --output-dir runs/sft-dev/paper
```

This uses the existing evaluator, resource sampler and official scorer. It
publishes completed episodes during evaluation, then the full fixed-split
report; partial running means are progress indicators, not headline results.
It requires a new output directory and does not selectively retry patients.
It always uses the simulator's paper prompt; it has no prompt-selection option.
`--seed N` overrides the sampling seed for a replicate (the config's seed is 0).
`--split train` evaluates the 680 training patients to measure how well a
checkpoint fits its own training data; that is not a generalization result.
Heldout is not selectable. `--mode offline` keeps a run out of the hosted project.
`tracking.py REPORT --traces TRACES --config CONFIG` backfills a completed run
without generating or rescoring answers. Keep locally saved artifacts regardless
of the hosted tracker.

## Rental shutdown

Run `training/rental_guard.py --deadline <ISO-8601-with-timezone>` under
Supervisor to enforce an absolute rental deadline across SSH disconnects.
It uses Vast's injected `CONTAINER_ID` / `CONTAINER_API_KEY`, never an
account-wide key, and requests **stop**, not destroy, through Vast's API.
It checks the response's `success` field: the instance's bundled CLI can exit
zero even when a stop is rejected. Transport/provider failures exit nonzero
without printing credentials. Store the chosen deadline in the Supervisor
command; restarting that process must not extend it.
Before resuming a later session, renew/remove the old deadline configuration
so an expired guard does not immediately stop the instance again.

For an early shutdown, copy results off the instance, disable the guard's
Supervisor configuration, then use the same script with a deadline of now.
Stopping preserves the disk but does not eliminate storage charges.
