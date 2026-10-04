# Project log

Running record of observations, decisions, and progress. Newest at the bottom. Times are local (EDT).
Entries before 2026-10-01 22:53 were reconstructed from the session and file timestamps, so their times are approximate.

## 2026-09-30

- **21:02** Plan saved as `PLAN.md`. Decision: primary model `Qwen/Qwen3-4B-Instruct-2507` (Prime-RL ships a tool-use RL example on it); Qwen3.5-4B sanity check; Qwen3.5-9B escalation only.
- **21:03** Cloned Synthetic Hospital v1.3 at `911f34c` into `external/` (gitignored, pinned in `scripts/setup_env.sh`).
- **21:05** Upstream compose failed: Docker did not share the repo path for the DB bind mount. Wrote `docker/compose.yaml` using the `with-data` image, simulator on port 58000.
- **21:06** Observation: reward depends on ICD-10-CM titles downloaded at container boot (chart-neutral sets fuzzy-match against them). Kept the download on; `shrl verify` checks it.
- **21:09** Dev split written: 680 train / 120 dev from upstream `train`, stratified, seed 20260930. `heldout` sealed (not reachable from CLI).
- **21:10** `shrl verify`: oracle = 1.000 on all 1,000 train + public instances. No reference ICD codes in observations.
- **21:10** Upstream bug: scorer returns HTTP 500 on submissions with string lists instead of scoring 0; episode left open. Harness treats it as malformed, reward 0.
- **21:10** Upstream bug: `view_section(section_id)` returns hidden assessment/plan sections. Low value (question-stem residue, no extra diagnosis names).
- **21:10** `shrl audit` (dev). **Main finding:** 81% of reference diagnosis names are carried verbatim into later notes' "Active Problem List". Copying dated entries with correct codes scores 0.859; ceiling without diagnosing the final visit is 0.880.
- **21:10** Audit: no padding or prior exploit (top-k categories < 0.03; +5 wrong codes halves reward). Predicted acuity has no effect on reward. Chart-neutral extra codes are free.
- **21:14** Audit on train (n=680): copy strategy 0.818.
- **21:20** OpenRouter does not serve Qwen3-4B-Instruct-2507 or Qwen3.5-4B. Decision: run Qwen3.5-9B and Qwen3-30B-A3B-Instruct-2507 as reference models.
- **21:45** Qwen3.5-9B with default reasoning is ~10x slower and mixes provider quantizations. Decision: abandon that run (55/120), rerun with reasoning off on bf16 providers.
- **21:50** Harness change: episodes killed by provider errors (429s) are retried, not scored.
- **22:13** Baselines on dev, paper prompt: Qwen3.5-9B (no-think) **0.415**; Qwen3-30B **0.463**. Both submit in ~100% of episodes; pipeline healthy.
- **22:13** Observation: agents read 80–94% of encounters but recover only ~half of the copyable diagnoses. Misses are wrong ICD code for a correct name (17–27%) or name seen but not submitted (24–32%). Final-visit recall ~20%.
- **22:14** Wrote `results/milestone1.md`. Decision rule "mostly solved by trivial extraction" is triggered.

## 2026-10-01

- **22:20** Tried the 4B model locally (LM Studio, MLX 8-bit). Loaded with 32k context and 4 parallel slots.
- **22:31** The local run crashed the machine (memory: 16 GB, model + large context + Docker). Decision: no local inference; 4B baseline waits for a rented GPU with vLLM and the exact checkpoint.
- **22:40** Decision (framing): option A. Stay on the official benchmark and metric; add an analysis-only column splitting recall into earlier visits (copyable) and final visit (must be reasoned). Rejected masking the problem list (leaves the published benchmark).
- **22:45** Docker Desktop is gone; OrbStack is the runtime now. Rebuilt the simulator; ICD-10-CM loaded (97,584 rows).
- **22:47** Added `shrl/breakdown.py`. Re-scored paper-prompt runs: recall earlier/final = 0.470/0.171 (9B), 0.487/0.200 (30B).
- **22:48** Wrote `prompts/strong_v1.md`: tells the model to copy dated problem-list entries, diagnose the final visit, code carefully, avoid extras.
- **22:53** Strong prompt, Qwen3-30B, dev: **0.439** (paper prompt 0.463). No improvement; within noise. Needs trajectory inspection before concluding anything. 9B strong-prompt run in progress.
- **22:53** Started this log at the user's request.
- **23:04** Strong prompt did not change 30B behaviour where it matters: recall earlier/final 0.493/0.214 (paper prompt 0.487/0.200), precision fell 0.544 to 0.457, actions fell 11.2 to 7.7. Inspected episodes: the model sees "X (diagnosed date)" and still submits its own guesses plus undated history entries. Instruction-following, not chart access, is the limit.
- **23:05** 9B strong-prompt run invalid: 114/120 episodes hit provider 429 rate limits (only 6 scored). Retrying at concurrency 3 with more retries. No 9B strong-prompt number yet.
- **23:06** Case study (gt_id 7454, 30B): both prompts scored 0. Model read all 3 notes, ignored the dated problem-list entries, missed neuroleptic malignant syndrome. Also observed label noise: reference name "Neuroleptic-induced akathisia" carries code G21.11 (neuroleptic-induced parkinsonism), so copying the name and coding it correctly (G25.71) would still miss. The 0.86 copy baseline relies on the benchmark's own name-to-code table; need to measure how often names and codes disagree.
- **23:19** 9B strong-prompt retry still rate-limited (10/120 after ~30 min). Left running; not blocking.
- **23:19** Coding-ceiling experiment (`scripts/coding_ceiling.py`, dev): give a model only the dated problem-list names, no chart, ask for ICD-10 codes. Reward: lookup table 0.859; Qwen3-235B 0.685 (agrees with reference category on 77% of names); Qwen3-30B 0.532 (57%). 30B full agent scores 0.463, i.e. below its own copy-and-code ceiling.
- **23:19** Conclusion: gap decomposes into (a) not copying listed diagnoses, ~0.07 for 30B; (b) ICD-10 coding of known names, the largest part, 0.53 to 0.86, partly benchmark-specific conventions and label noise; (c) final-visit diagnosis, recall ~0.2. Training gains will come mostly from (b). Seen-vs-unseen diagnosis analysis is therefore essential (62% of dev diagnoses appear in train).
- **23:19** Decision (user): prompting does not fix it; treat as a positive finding and move on to the training phases.
- **23:23** User paused the move to training: "throw RL at it" is not a plan. Reassessment (PLAN decision rule on trivial extraction): on `patient_diagnosis` the model already reads the chart; the gap is copying, ICD-10 coding, and final-visit diagnosis. All three have labels in train, and the efficient tool-use process is two calls, so SFT gets everything RL would. Expected outcome if we train here: SFT large gain, RL about equal. Open decision: train on diagnosis anyway, or first audit the benchmark's other official tasks for one where chart interaction is the bottleneck. No GPU or training work started.
- **23:50** Decision (user): Path 2. Audit the other official tasks before any training. User is also fine treating the copy finding as a write-up caveat on the benchmark (option C); the visit-split column stays as analysis only.

## 2026-10-02

- **00:05** Audited the other three official tasks on dev (`scripts/audit_other_tasks.py`, `results/milestone2_other_tasks.md`).
- **00:05** `evidence_retrieval`: relevance labels are a rule on section type. Sorting sections by type scores P@5 0.98; 5 random sections score 0.71. Not usable.
- **00:05** `context_summarization` (whole patient): reward is recall only. Pasting the entire chart scores 0.70; longer is always better. Reward exploitable; not usable without changing it.
- **00:05** `imaging_indication`: no shortcut, but little headroom. 30B: no chart 0.327, current note 0.410, full chart 0.409, agent 0.374. 235B with note 0.453. Prior visits add nothing. Remaining gap to 1.0 is overlap with an LLM-written reference's wording.
- **00:05** Conclusion: in none of the four tasks is chart interaction the bottleneck. Fall back to `patient_diagnosis` (Path 1), now with that checked. Benchmark-wide finding goes in the write-up.
- **00:05** 9B strong-prompt retry abandoned: provider still rate-limiting (18 scored of 120). No 9B strong-prompt number; the 30B result stands alone.
- **00:31** GPU price check (provider pages, 2026-10-02): 48 GB cards on-demand per hour: RunPod A6000 $0.33 community / $0.53 secure, A40 $0.35 / $0.49; Prime Intellect A6000 ~$0.47 (third-party tracker); Lambda A6000 $1.09. Saturn lists no 48 GB card (A10G 24 GB $1.50). Colab ruled out (no SSH, sessions die, no Docker). Leaning RunPod; pods are containers, so the simulator would run natively (Postgres + Redis installed in the pod) instead of via Docker. Awaiting user choice.
- **00:35** Checked Prime Intellect Hosted Training (`prime train run <config.toml>`). Docs: shared hosted LoRA training stops accepting new runs on 2026-10-05; RL only (no SFT); priced per token; current price list shows Qwen3.5-4B but not Qwen3-4B-Instruct-2507; docs silent on whether the environment can reach an external simulator. Recommendation: rent the A6000 through Prime Intellect and run open-source prime-rl on it ourselves. Awaiting user.
- **21:04** Decision (user): working mode. Claude builds everything; user owns understanding of the core pieces (environment, reward, training config, SFT data, trajectories) through short walkthroughs, written predictions before each run, and small hands-on changes. Starting no-GPU work while the A6000 is provisioned: Verifiers environment wrapper for `/env`, then SFT demonstration generator.
- **21:08** Updated the global uv in ~/.local/bin from 0.5.24 to latest: the Verifiers null harness runs `uv sync --script` and puts ~/.local/bin first on PATH. Reversible with `uv self update 0.5.24`.
- **21:15** Built Verifiers v1 taskset `environments/synthetic_hospital_dx` (MCP toolset forwarding the paper's 10 tools to `/env/step`; reward = official per-episode score; stop on submit). Smoke test through `vf-eval` with OpenRouter 30B: 3 episodes ran end to end (rewards 1.0, 0.64, 0.0). Two behaviour differences from `shrl/agent.py`: malformed tool arguments are rejected before reaching the simulator (no action spent), and a turn with no tool call ends the episode (no nudge).
- **21:15** Built SFT demonstrations `data/sft/oracle-copy-v1` (680 train patients): real episodes of view_encounters -> latest note -> submit the reference answer. All score 1.0; ~3–5k tokens each. Design choice open for the user: the demos submit labels the note does not fully reveal (final-visit diagnosis).
- **21:15** Correction: prime-rl's `rl` launcher needs 2 GPUs (1 inference + 1 trainer); it does not share one card. Baseline eval and SFT run on 1 GPU. Earlier advice ("one 48 GB card for everything") was wrong for RL.
- **21:23** Verifiers dev run, 30B: **0.263** vs 0.463 in `shrl/agent.py`. Cause: 47/120 episodes ended without submitting; in 46 the model wrote its answer as JSON text instead of calling the tool. Verifiers' default ends the rollout on a tool-less reply; the paper's harness nudges ("Please take an action...") and ends with a forced-submission turn that parses a JSON answer from text.
- **21:23** Fix: `environments/.../env.py` ports the paper's control flow (nudge while turns remain, then one forced turn, text answers parsed and submitted). New metrics `nudges`, `submitted_as_text`. Smoke test: 8/8 episodes submitted. Also moved tool argument docs into parameter descriptions so tool descriptions match the paper's. Remaining known difference: the submit tool's schema uses JSON-schema $defs references instead of the paper's inline objects.
- **21:23** Regenerated `data/sft/oracle-copy-v1` with the updated tool schemas. Full dev rerun of 30B running.
- **21:31** Corrected Verifiers dev run, 30B: **0.466** (95% CI 0.406–0.531), vs 0.463 in `shrl/agent.py`; paired difference +0.003 (CI −0.043 to +0.051). 120/120 submitted; 28 episodes needed a nudge; none needed the text fallback. The Verifiers pipeline matches the earlier harness and is the reference from here on.

## 2026-10-04

- **12:25** Reorganized the repo by phase (user request: clear for new readers before renting hardware). Checkpoint commit first (b5b3d62), then a pure-move commit. `shrl/` is gone: shared code in `common/` (simulator client, benchmark reader, split, metrics incl. the visit split); Phase 1–2 checks in `audit/`; the early OpenRouter harness and milestone-1 report in `baselines/`; the Verifiers environment, SFT builder and run configs in `training/`; server in `hospital/`; split and SFT data in `data/`; raw results in `results/raw/<phase>/`. No central CLI: `uv run python -m <folder>.<module>`; each folder has a README. Deleted: smoke-test result files and the invalid rate-limited 9B strong-prompt result (6 scored episodes).
- **12:25** Checks after the move: unit tests pass; `audit.strategies --split dev` reproduces every number; `tables.md` identical except the deleted invalid row; split and SFT data regenerate byte-identical; 3-episode `vf-eval` smoke run and 1-episode `baselines.agent` run both work.
