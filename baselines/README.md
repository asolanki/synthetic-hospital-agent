# baselines/

This is my first agent harness, written before I switched to the Verifiers environment in [`training/`](../training/README.md). It plays the exact same episode (same server, prompt, tools, and action budget) and gets the same score, but the RL trainer can't train through it. So for anything new, use `training/`. I kept this folder so the earliest baseline numbers still reproduce.

| Command | What it does |
|---|---|
| `uv run python -m baselines.agent --model <openrouter id> --split dev` | Runs one episode per validation patient. Transcripts go to `runs/<name>/`, the summary to `results/raw/baselines/<name>.json` |
| `uv run python -m baselines.agent --model <id> --prompt-file baselines/prompts/strong_v1.md --name <name>` | Same thing with my "stronger" step-by-step prompt. It didn't help |
| `uv run python -m baselines.report` | Rebuilds the baseline tables from `results/raw/` |
