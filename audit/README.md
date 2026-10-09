# audit/ — Phases 1–2: is the benchmark sound, and what does it measure?

Done. Findings: [results/milestone1.md](../results/milestone1.md) and
[results/milestone2_other_tasks.md](../results/milestone2_other_tasks.md); the single-turn Gemma
reproduction and the released scorer's chart-neutral gap are in
[results/milestone5_head_to_head.md](../results/milestone5_head_to_head.md).
Raw outputs: `results/raw/audit/` and `results/raw/other_tasks/`.

| Command | What it answers | Output |
|---|---|---|
| `uv run python -m audit.verify` | Does the oracle score 1.0, do malformed answers score 0, does any tool leak the answer key? | `raw/audit/verify.json` |
| `uv run python -m audit.strategies --split dev` | How far do hand-written strategies get (copy the problem list, guess common codes, pad the answer)? | `raw/audit/strategies_dev.json` |
| `uv run python -m audit.coding_ceiling --model <id>` | Given only the copyable names, how well can a model code them? | `raw/audit/coding_ceiling_<model>_dev.json` |
| `uv run python -m audit.other_tasks <task>` | Same questions for the benchmark's three other tasks | `raw/other_tasks/*.json` |
| `docker exec … python /tmp/harness_gap.py` (inside the simulator container; usage in its docstring) | Why the paper's agents score far higher. Its harness leaks coded reference diagnoses, a known issue in upstream's README; this measures how much, with and without the leak, against our `/env` harness | `raw/audit/harness_gap_dev.json` |
| `uv run vf-eval @ audit/harness_gap_env.toml -n 120 -c 8 --no-rich --output-dir runs/harness-check/env-qwen30b-t0 --run.dir eval`, then `uv run python -m training.report runs/harness-check/env-qwen30b-t0/eval/traces.jsonl --name qwen3-30b-env-dev-t0` | The `/env` arm of the harness-gap audit: Qwen3-30B in our Verifiers harness at the paper adapter's controls (temperature 0, 16,384 max tokens), 120 dev patients | `raw/training/qwen3-30b-env-dev-t0.json` |
| `docker exec … python /tmp/paper_singleturn.py` (inside the simulator container; usage in its docstring) | Does the paper's own single-turn pipeline (locked CoT prompt, its parser and scorer) reproduce Table 3's Gemma 3 27B patient-diagnosis F1 of 0.287? Reports the paper's batch F1 and a strict F1 that scores failures as zero | `raw/audit/paper_singleturn_gemma3_public.json` |
| Label audit (protocol `label_audit_protocol.md`; results `label_audit_results.md`; spot-check form `label_audit_spotcheck.md`, pending): `label_scan.py` (deterministic scan against the public code book), `label_audit_build_candidates.py`, LLM judges per `label_audit_judge_instructions.md`, then `label_audit_finalize.py` | How many reference codes are wrong at the scoring category, and does correcting them change any result? About 5.7% (CI 4.1–7.9%); no verdict changes. Run in a cloud workspace and saved as run: `finalize.py` has that workspace's paths and imports the rescorer as `rescore` | `raw/audit/label_audit_{candidates,verdicts,summary}.json`, `label_audit_wrong_labels.csv` (the list for the authors) |
| `uv run python audit/label_audit_rescore.py external/synthetic_hospital external/synthetic_hospital/benchmark_v1.3.db runs/icd10cm/icd10cm_tabular_2025.xml results/raw/training <report names…>` | Rescore saved public submissions with upstream's own scorer and chart-neutral sets (it reproduces recorded rewards exactly), optionally with corrected labels | printed; told-vs-taught arms in `raw/audit/label_audit_rescore_told_arms.json` |

`coding_ceiling`, `other_tasks` and `harness_gap` call OpenRouter (key in `.env`). `other_tasks imaging-agent`
reuses the early harness in `baselines/agent.py`. `paper_singleturn` calls DeepInfra directly (`DEEPINFRA_API_KEY`).
