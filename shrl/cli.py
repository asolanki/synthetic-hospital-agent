"""Command line: `shrl verify | split | audit | baseline | report`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from shrl import config


def _load_dotenv() -> None:
    path = config.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _write_result(name: str, payload: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def cmd_verify(args) -> int:
    from shrl import verify
    result = asyncio.run(verify.run(crawl_n=args.crawl))
    _write_result("verify", result)
    print(json.dumps({k: v for k, v in result.items() if k != "budget"}, indent=1))
    if result["failures"]:
        print("\nFAILED:", *result["failures"], sep="\n  - ")
        return 1
    print("\nall hard checks passed")
    return 0


def cmd_split(args) -> int:
    from shrl import splits
    split = splits.write()
    print(json.dumps({k: v for k, v in split.items() if k not in ("train", "dev")}, indent=1))
    print(f"wrote {config.SPLIT_FILE}")
    return 0


def cmd_audit(args) -> int:
    from shrl import audit, data, report, splits
    split = splits.load()
    by_id = {i.gt_id: i for i in data.load_instances(("train",))}
    result = asyncio.run(audit.run([by_id[g] for g in split[args.split]], [by_id[g] for g in split["train"]]))
    result["split"] = args.split
    _write_result(f"audit_{args.split}", result)
    print(report.audit_table(result))
    return 0


def cmd_baseline(args) -> int:
    from shrl import agent, report, splits
    gt_ids = splits.gt_ids(args.split)
    if args.limit:
        gt_ids = gt_ids[:args.limit]
    cfg = agent.RolloutConfig(
        model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
        budget=args.budget, seed=args.seed,
        system_prompt=(config.ROOT / args.prompt_file).read_text().strip() if args.prompt_file else None,
        extra_body=json.loads(args.extra_body) if args.extra_body else None)
    name = args.name or f"{args.model.replace('/', '_').replace(':', '_')}-{args.split}"
    summary = asyncio.run(agent.run(cfg, gt_ids, config.RUNS_DIR / name, concurrency=args.concurrency))
    summary.update(name=name, split=args.split)
    _write_result(f"baseline_{name}", summary)
    print(report.baseline_table([summary]))
    return 0


def cmd_sft_data(args) -> int:
    from pathlib import Path

    from shrl import sft_data
    meta = asyncio.run(sft_data.build(args.name, Path(args.tools_from),
                                      Path(args.system_prompt) if args.system_prompt else None, args.split))
    print(json.dumps(meta, indent=1))
    return 0


def cmd_report(args) -> int:
    from shrl import report
    print(report.build())
    return 0


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="shrl")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify", help="Phase 1: scorer sanity and label-exposure checks")
    p.add_argument("--crawl", type=int, default=40, help="episodes to crawl for label exposure")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("split", help="write the fixed train/dev split")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("audit", help="Phase 2: score hand-written strategies")
    p.add_argument("--split", choices=["dev", "train"], default="dev")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("baseline", help="roll out a chat model in the agent environment")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--split", choices=["dev", "train", "public"], default="dev")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--budget", type=int, default=config.DEFAULT_BUDGET)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-body", default=None, help="JSON merged into the chat request body")
    p.add_argument("--prompt-file", default=None,
                   help="system prompt file relative to the repo root (default: the environment's paper prompt)")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("sft-data", help="write scripted demonstrations for SFT")
    p.add_argument("--name", required=True)
    p.add_argument("--tools-from", required=True, help="a Verifiers traces.jsonl from synthetic-hospital-dx")
    p.add_argument("--system-prompt", default=None, help="prompt file; default is the paper prompt")
    p.add_argument("--split", choices=["train"], default="train")
    p.set_defaults(func=cmd_sft_data)

    p = sub.add_parser("report", help="rebuild results/*.md from results/*.json")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
