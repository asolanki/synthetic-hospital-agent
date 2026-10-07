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
| `docker exec … python /tmp/harness_gap.py` (inside the simulator container; usage in its docstring) | Why the paper's agents score far higher: its harness leaks coded reference diagnoses; measured with and without the leak against our `/env` harness | `raw/audit/harness_gap_dev.json` |
| `docker exec … python /tmp/paper_singleturn.py` (inside the simulator container; usage in its docstring) | Does the paper's own single-turn pipeline (locked CoT prompt, its parser and scorer) reproduce Table 3's Gemma 3 27B patient-diagnosis F1 of 0.287? Reports the paper's batch F1 and a strict F1 that scores failures as zero | `raw/audit/paper_singleturn_gemma3_public.json` |

`coding_ceiling`, `other_tasks` and `harness_gap` call OpenRouter (key in `.env`). `other_tasks imaging-agent`
reuses the early harness in `baselines/agent.py`. `paper_singleturn` calls DeepInfra directly (`DEEPINFRA_API_KEY`).
