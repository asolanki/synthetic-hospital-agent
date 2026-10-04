"""Fixed train/dev split of the released training patients.

The 800 `train` patients are partitioned into train and dev, stratified by
(dominant ICD-10 chapter of the reference, encounter-count bucket) with a fixed
seed, mirroring how upstream drew `heldout` from the same pool. `heldout` (268
patients) is the sealed final evaluation set and never appears in this file;
`public` (200 patients) is the paper's reported benchmark and is kept for
comparison against published numbers.
"""

from __future__ import annotations

import collections
import json
import random

from shrl import config, data

SEED = 20260930
DEV_SIZE = 120


def _bucket(n: int) -> str:
    return "2-3" if n <= 3 else "4-5" if n <= 5 else "6+"


def _stratum(inst: data.Instance) -> tuple[str, str]:
    chapters = collections.Counter((d.get("icd10") or "?")[0] for d in inst.gt_diagnoses)
    dominant = sorted(chapters.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if chapters else "?"
    return dominant, _bucket(len(inst.encounters))


def build(dev_size: int = DEV_SIZE, seed: int = SEED) -> dict:
    pool = data.load_instances(("train",))
    strata: dict[tuple[str, str], list[data.Instance]] = collections.defaultdict(list)
    for inst in pool:
        strata[_stratum(inst)].append(inst)

    rng = random.Random(seed)
    quota = dev_size / len(pool)
    dev: set[int] = set()
    carry = 0.0
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda i: i.gt_id)
        rng.shuffle(members)
        want = len(members) * quota + carry
        k = int(round(want))
        carry = want - k
        dev.update(i.gt_id for i in members[:k])

    train = [i for i in pool if i.gt_id not in dev]
    dev_insts = [i for i in pool if i.gt_id in dev]
    train_dx = {d["diagnosis_id"] for i in train for d in i.gt_diagnoses}
    train_cat = {data.category(d["icd10"]) for i in train for d in i.gt_diagnoses}
    dev_dx = [d for i in dev_insts for d in i.gt_diagnoses]
    return {
        "task": config.TASK,
        "seed": seed,
        "source_split": "train",
        "stratified_by": ["dominant_icd10_chapter", "encounter_count_bucket"],
        "train": sorted(i.gt_id for i in train),
        "dev": sorted(dev),
        "stats": {
            "n_train": len(train), "n_dev": len(dev),
            "dev_diagnoses": len(dev_dx),
            "dev_diagnosis_seen_in_train": sum(d["diagnosis_id"] in train_dx for d in dev_dx) / len(dev_dx),
            "dev_category_seen_in_train": sum(data.category(d["icd10"]) in train_cat for d in dev_dx) / len(dev_dx),
        },
    }


def write() -> dict:
    split = build()
    config.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.SPLIT_FILE.write_text(json.dumps(split, indent=1) + "\n")
    return split


def load() -> dict:
    if not config.SPLIT_FILE.exists():
        raise FileNotFoundError(f"{config.SPLIT_FILE} missing; run `shrl split`")
    return json.loads(config.SPLIT_FILE.read_text())


def gt_ids(name: str) -> list[int]:
    """gt_ids for `train`, `dev`, or `public`. The sealed heldout split is not reachable here."""
    if name in ("train", "dev"):
        return load()[name]
    if name == "public":
        return [i.gt_id for i in data.load_instances(("public",))]
    raise ValueError(f"unknown or sealed split '{name}' (choose train, dev, or public)")
