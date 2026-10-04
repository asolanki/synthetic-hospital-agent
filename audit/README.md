# audit/ — Phases 1–2: is the benchmark sound, and what does it measure?

Done. Findings: [results/milestone1.md](../results/milestone1.md) and
[results/milestone2_other_tasks.md](../results/milestone2_other_tasks.md).
Raw outputs: `results/raw/audit/` and `results/raw/other_tasks/`.

| Command | What it answers | Output |
|---|---|---|
| `uv run python -m audit.verify` | Does the oracle score 1.0, do malformed answers score 0, does any tool leak the answer key? | `raw/audit/verify.json` |
| `uv run python -m audit.strategies --split dev` | How far do hand-written strategies get (copy the problem list, guess common codes, pad the answer)? | `raw/audit/strategies_dev.json` |
| `uv run python -m audit.coding_ceiling --model <id>` | Given only the copyable names, how well can a model code them? | `raw/audit/coding_ceiling_<model>_dev.json` |
| `uv run python -m audit.other_tasks <task>` | Same questions for the benchmark's three other tasks | `raw/other_tasks/*.json` |

`coding_ceiling` and `other_tasks` call OpenRouter (key in `.env`). `other_tasks imaging-agent`
reuses the early harness in `baselines/agent.py`.
