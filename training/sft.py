"""LoRA SFT of the pinned checkpoint on the existing train-only demonstrations.

Uses the native inference chat template, not a second renderer. Only sampled
assistant spans receive loss: role openers, tool observations and template
newlines after end-of-turn are masked. Overlong examples fail, never truncate.

    bash training/run_sft.sh --prepare-only --output-dir runs/sft-preparation
    bash training/run_sft.sh --max-steps 2 --no-merge --output-dir runs/sft-setup-check
    bash training/run_sft.sh
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time
import tomllib

from common import config


def load_training_rows(path: Path, train_ids: set[int], dev_ids: set[int]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["gt_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != train_ids or set(ids) & dev_ids:
        raise ValueError("SFT data must contain each fixed train patient exactly once, with no dev patients")
    return rows


def encode_demo(row: dict, tokenizer, max_length: int) -> dict:
    messages = row["messages"]
    tools = json.loads(row["tools"])
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_indices:
        raise ValueError("Demonstration has no assistant targets")
    # Scorer output after the last assistant turn cannot supervise an earlier token.
    messages = messages[:assistant_indices[-1] + 1]
    ids = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=False)
    if len(ids) > max_length:
        raise ValueError(f"gt_id={row['gt_id']}: {len(ids)} tokens exceeds max_length={max_length}; not truncating")
    labels = [-100] * len(ids)
    for i in assistant_indices:
        prompt = tokenizer.apply_chat_template(
            messages[:i], tools=tools, tokenize=True, add_generation_prompt=True)
        completed = tokenizer.apply_chat_template(
            messages[:i + 1], tools=tools, tokenize=True, add_generation_prompt=False)
        if completed[:len(prompt)] != prompt or ids[:len(completed)] != completed:
            raise ValueError("Chat template is not prefix-stable; cannot safely assign assistant loss")
        end = len(completed)
        while end > len(prompt) and completed[end - 1] != tokenizer.eos_token_id:
            end -= 1
        if end <= len(prompt):
            raise ValueError("Assistant turn has no terminating EOS token")
        labels[len(prompt):end] = ids[len(prompt):end]
    return {"input_ids": ids, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config.ROOT / "training/configs/sft_oracle_copy.toml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, help="Positive optimizer-step limit for the training setup check")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-merge", action="store_true", help="Save the adapter without a merged inference checkpoint")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    settings = tomllib.loads(args.config.read_text())
    output = (args.output_dir or config.ROOT / settings["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path = config.ROOT / settings["data"]
    split = json.loads(config.SPLIT_FILE.read_text())
    rows = load_training_rows(data_path, set(split["train"]), set(split["dev"]))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"], revision=settings["revision"], local_files_only=True)
    encoded = [encode_demo(row, tokenizer, settings["max_length"]) for row in rows]
    lengths = [len(e["input_ids"]) for e in encoded]
    targets = [sum(label != -100 for label in e["labels"]) for e in encoded]
    manifest = {
        "config": settings, "max_steps_override": args.max_steps, "wandb_mode": args.wandb_mode,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(config.SPLIT_FILE.read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_gt_ids": sorted(row["gt_id"] for row in rows), "examples": len(rows),
        "input_tokens": {"total": sum(lengths), "median": statistics.median(lengths), "max": max(lengths)},
        "assistant_target_tokens": {"total": sum(targets), "median": statistics.median(targets), "max": max(targets)},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate", "datasets", "wandb")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("examples", "input_tokens", "assistant_target_tokens", "versions")}), flush=True)
    if args.prepare_only:
        return

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("SFT requires the CUDA GPU; stop the inference service before training")
    set_seed(settings["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        settings["model"], revision=settings["revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"], target_modules="all-linear", bias="none",
        revision=settings["revision"],
    ))
    trainable, total = model.get_nb_trainable_parameters()
    manifest.update(trainable_parameters=trainable, total_parameters=total,
                    cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with wandb.init(
        project=settings["wandb_project"], entity=settings["wandb_entity"],
        name=output.name, group="qwen3-4b",
        job_type="sft-setup-check" if args.max_steps is not None else "sft",
        config=manifest, dir=str(output), mode=args.wandb_mode,
        settings=wandb.Settings(disable_git=True, save_code=False),
    ) as run:
        manifest["wandb_run_id"] = run.id
        manifest["wandb_url"] = run.url
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"wandb_url": run.url}), flush=True)
        trainer = Trainer(
            model=model, processing_class=tokenizer, train_dataset=Dataset.from_list(encoded),
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
            args=TrainingArguments(
                output_dir=str(output / "checkpoints"), num_train_epochs=settings["epochs"],
                max_steps=args.max_steps if args.max_steps is not None else -1,
                per_device_train_batch_size=settings["micro_batch_size"],
                gradient_accumulation_steps=settings["gradient_accumulation_steps"],
                learning_rate=settings["learning_rate"], warmup_ratio=settings["warmup_ratio"],
                weight_decay=settings["weight_decay"], max_grad_norm=settings["max_grad_norm"],
                lr_scheduler_type="cosine", optim="adamw_torch", bf16=True, tf32=True,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                seed=settings["seed"], data_seed=settings["seed"],
                logging_steps=1, logging_nan_inf_filter=False, disable_tqdm=True,
                # A sweep keeps every epoch's checkpoint so each can be evaluated; default keeps one.
                save_strategy="epoch", report_to=["wandb"],
                save_total_limit=None if settings.get("keep_epoch_checkpoints") else 1,
                run_name=output.name, label_names=["labels"],
            ),
        )
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        outcome = trainer.train()
        trainer.save_model(str(output / "adapter"))
        trainer.save_state()
        result = {
            **outcome.metrics, "wall_seconds": time.monotonic() - started,
            "optimizer_steps": trainer.state.global_step,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "adapter": str(output / "adapter"), "wandb_url": run.url,
        }
        if not args.no_merge:
            merged = model.merge_and_unload(safe_merge=True)
            merged.config.use_cache = True
            merged.save_pretrained(output / "merged", safe_serialization=True)
            tokenizer.save_pretrained(output / "merged")
            result["merged_checkpoint"] = str(output / "merged")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        run.summary.update(result)
        artifact = wandb.Artifact(output.name, type="model", metadata={
            "base_model": settings["model"], "base_revision": settings["revision"],
            "data_sha256": manifest["data_sha256"],
        })
        artifact.add_dir(str(output / "adapter"), name="adapter")
        for path in (args.config, Path(__file__), output / "manifest.json", output / "result.json",
                     output / "checkpoints/trainer_state.json"):
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
