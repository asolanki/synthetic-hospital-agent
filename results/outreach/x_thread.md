# X thread draft (not posted)

Voice matched to @adarshsolanki's recent posts. Numbers are official F1 on the paper's 200 public patients unless stated, and match the blog (`rl-medical-coder.md`). Sources: `results/milestone3_training.md`, `results/milestone4_sft_and_codebook.md`, `results/raw/training/*-env-public*.json`, `results/raw/audit/harness_gap_dev.json`, `results/raw/audit/paper_singleturn_gemma3_public.json`, `LOG.md` 2026-10-07. Character counts treat each link as 23. Replace `[link]` with the blog URL.

## 1/15 (234 chars)

spent a week trying to get RL to teach a 4b model to read patient charts, on CMU's new Synthetic Hospital benchmark.

RL did basically nothing. a free ICD-10 lookup tool + plain SFT got the 4b level with kimi k2.5 (~1T params). thread

## 2/15 (234 chars)

setup: qwen3-4b gets a patient id and 10 EHR tools, reads a multi-visit chart, submits a problem list with ICD-10 codes. scored by severity-weighted F1 on the paper's 200 public patients.

all training is LoRA on 680 training patients

## 3/15 (233 chars)

first, what it measures: later notes list earlier diagnoses verbatim, by design. copy that list and code it with the benchmark's own table: 0.82 F1.

so it's mostly reading + ICD-10 coding. only the final visit needs a real diagnosis

## 4/15 (205 chars)

round 1, RL:
base: 0.17
SFT (3 epochs): 0.39
SFT + GRPO: 0.40

GRPO only reinforces answers the model already samples. a third of diagnoses were never coded right in any of 4 samples. nothing to learn from

## 5/15 (166 chars)

round 2, the boring fix: the SFT was undertrained. swept 1-10 epochs, picked by dev.

8 epochs: 0.53 on public, +0.13 over RL. you don't need RL, you need more epochs

## 6/15 (205 chars)

where's the error? the model writes down basically every diagnosis in the chart, then gets the ICD-10 code wrong.

it's memorization: codes it saw in 5+ training patients, 84% right. codes it never saw: 1%

## 7/15 (234 chars)

so give it the book. CMS publishes ICD-10-CM for free. BM25 over the official titles, index and tables, exposed as one tool.

no benchmark answers in it, just what a human coder looks things up in. 16 MB index, ~5 ms per search on CPU

## 8/15 (233 chars)

tools must be taught. tool alone, untrained 4b: 0.17 → 0.49.

the 8-epoch SFT model ignored the tool completely (it had learned a fixed 3-call routine). same 680 demos + one lookup step per diagnosis: 0.74. never-seen codes: 1% → 57%

## 9/15 (232 chars)

head-to-head: same harness, same 200 patients. the big models get no tool and no fine-tuning, that's the point.

4b, SFT + tool: 0.74
kimi k2.5 (~1T): 0.73
gpt-6 luna: 0.69
gemma 3 27b (paper's smallest): 0.26
same 4b, no tool: 0.53

## 10/15 (242 chars)

paired over the same patients:
vs luna: +0.07 (CI +0.03 to +0.11), a win
vs kimi: +0.03, CI crosses zero, so level
vs gemma: +0.49

without the tool the same 4b trails kimi and luna by 0.13-0.17. the tool is what moves it from behind to level

## 11/15 (267 chars)

but they win different halves.

diagnoses the chart already documents (copy + code): 4b 85%, frontier 70%
the new final-visit diagnosis you have to reason out: frontier 58-65%, 4b 41%

a tool closes a knowledge gap, not a reasoning gap

next up: RL on the final visit

## 12/15 (252 chars)

and it runs on hardware you control: 8 GB model, 132 MB LoRA adapter, 16 MB index.

200 patients in ~3 min on one A100, ~$0.26 per 1,000 patients in GPU rent. kimi's API: ~$51. for real patient data, keeping it in-house probably matters more than price

## 13/15 (258 chars)

the paper's agents score 0.81-0.89, but the authors' README already flags that harness: two tools return the gold diagnoses with codes. i measured how much, qwen3-30b, same patients:
leaky: 0.73
leak masked: 0.40
clean harness: 0.42

~+0.33 is the answer key

## 14/15 (222 chars)

side quests:
- vllm truncated streamed tool-call JSON under load
- merging a small RL LoRA into bf16 erased ~99% of the update
- prime-rl deadlocked on shutdown
- gemma 3's tool calls needed a translator

total spend: ~$54

## 15/15 (268 chars)

to be clear the benchmark is genuinely good: realistic charts, open data, verifiable rewards, and its /env endpoints already fix the leak.

lesson: train longer, build the tool a human would use, then teach the model to use it.

writeup + code: [link]

## Tweet to the authors (272 chars)

Post as a reply to Christine Park's launch post (https://x.com/sparkcpark/status/2103474938095239336) once the thread is live.

@sparkcpark @valeriechen_ @Tim_Dettmers had a lot of fun building on synthetic hospital. a LoRA'd 4b + a free ICD-10 lookup tool gets level with kimi k2.5 in your clean /env harness. also measured the leak your README mentions + two scoring quirks: [thread link]
