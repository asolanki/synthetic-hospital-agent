# audit/

Before training anything, I wanted to know what this benchmark actually measures and whether it could be gamed. This folder is all of that poking around. The findings are in [milestone 1](../results/milestone1.md) and [milestone 2](../results/milestone2_other_tasks.md), and the raw outputs are in `results/raw/audit/` and `results/raw/other_tasks/`.

The benchmark server needs to be running for most of these (see [`hospital/`](../hospital/README.md)). The ones that call hosted models need `OPENROUTER_API_KEY` in `.env`.

| Command | What it answers |
|---|---|
| `uv run python -m audit.explore_data` | What's in the data? Chart sizes, how many diagnoses per patient, how rare the codes are. The numbers in the blog's "Looking at the data" section |
| `uv run python -m audit.verify` | Does the scorer work? A perfect answer should score 1.0 and an empty one 0. Also checks no tool leaks the answer key |
| `uv run python -m audit.strategies --split dev` | How far do dumb strategies get? Copying the problem list, guessing common codes, padding the answer |
| `uv run python -m audit.coding_ceiling --model <id>` | Given only the diagnosis names from the problem list, how well can a model code them? |
| `uv run python -m audit.other_tasks <task>` | The same questions for the benchmark's other three tasks |
| `python audit/harness_gap.py` (inside the server's container, usage in the file) | How much does the paper's agent harness leak? It hands the agent the coded answers. This measures it with and without the leak |
| `python audit/paper_singleturn.py` (inside the server's container) | Can I reproduce the paper's single-turn Gemma 3 27B score? (0.283 vs the paper's 0.287) |

## The label audit

About 6% of the answer key's codes look wrong. The full write-up is [`label_audit_results.md`](label_audit_results.md) and the list of wrong labels is `results/raw/audit/label_audit_wrong_labels.csv`.

How it worked:

1. `label_scan.py` flags (name, code) pairs where the code doesn't match the name in the official ICD-10-CM list.
2. `label_audit_build_candidates.py` builds the batches for review.
3. Two Claude Sonnet judges and a Claude Opus tiebreaker reviewed them, following [`label_audit_judge_instructions.md`](label_audit_judge_instructions.md).
4. `label_audit_finalize.py` combines the verdicts and rescores every run with the corrected labels (using `label_audit_rescore.py`, which runs the paper's own scorer).

The judging ran once, in a separate workspace, so the verdicts are saved in `results/raw/audit/` rather than regenerated here. [`label_audit_spotcheck.md`](label_audit_spotcheck.md) is 30 random verdicts for a human to check.
