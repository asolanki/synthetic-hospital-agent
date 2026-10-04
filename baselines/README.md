# baselines/ — Phase 3a: the first agent harness

Done. This is the harness that produced the milestone-1 baseline table, written before
the Verifiers environment existed. It plays the same episode as `training/environment/`
(same server, prompt, tools, budget and nudge rule) and gives the same score on the dev
patients (Qwen3-30B: 0.463 here, 0.466 there), but prime-rl cannot train through it.
**New runs go through `training/`.** This folder stays so the milestone-1 numbers can be
reproduced.

| Command | What it does |
|---|---|
| `uv run python -m baselines.agent --model <openrouter id> --split dev` | One episode per dev patient; trajectories to `runs/<name>/`, summary to `results/raw/baselines/<name>.json` |
| `uv run python -m baselines.agent --model <id> --prompt-file baselines/prompts/strong_v1.md --name <name>` | Same, with the strong prompt instead of the paper's |
| `uv run python -m baselines.report` | Rebuild `results/tables.md` from `results/raw/` |

| File | What it is |
|---|---|
| `agent.py` | The episode loop: model call → tool call → simulator step → repeat → submit |
| `report.py` | Markdown tables for milestone 1 |
| `prompts/strong_v1.md` | The "strong" system prompt (tells the model to copy dated problem-list entries). It did not help. |
