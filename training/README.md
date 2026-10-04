# training/ — Phases 3b–4: everything prime-rl uses

In progress. Evaluation, SFT and RL of the 4B model all run through the environment
package here, so they share one episode definition and one score.

| Path | What it is |
|---|---|
| `environment/` | The benchmark as a [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 package (`synthetic-hospital-dx`), which prime-rl loads by name. Standalone on purpose: it does not import `common/`. |
| `sft_data.py` | Builds SFT demonstrations into `data/sft/<name>/` |
| `configs/` | Run configs (`eval_openrouter.toml` now; SFT and RL configs next) |

## The environment package

| File | Role in an episode |
|---|---|
| `taskset.py` | One task per patient. `setup` opens a simulator episode; `finalize` closes it; the reward is the official per-episode score |
| `tools.py` | The paper's 10 EHR tools, served over MCP; each call is forwarded to `/env/step` |
| `env.py` | The paper's control flow: nudge after a reply with no tool call, then one forced-submission turn |
| `state.py` | Per-episode bookkeeping (episode id, actions, submitted, reward) |
| `simulator.py` | Small HTTP helper (its own copy of the server address and token) |

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
