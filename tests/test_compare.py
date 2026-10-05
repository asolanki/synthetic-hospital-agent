import json

import pytest

from training import compare


def _report(path, rewards, *, seed):
    rows = [{"gt_id": gt_id, "reward": reward, "ok": True} for gt_id, reward in rewards.items()]
    path.write_text(json.dumps({"split": "dev", "complete_split": True, "records": rows,
                                "summary": {"name": path.stem}, "sampling_seed": seed}))
    return path


@pytest.fixture
def dev_split(tmp_path, monkeypatch):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [1, 2]}))
    monkeypatch.setattr(compare.config, "SPLIT_FILE", split)


def test_replicates_are_averaged_per_patient_before_pairing(tmp_path, dev_split):
    reference = [_report(tmp_path / "sft.json", {1: 0.5, 2: 0.5}, seed=0)]
    candidate = [_report(tmp_path / "rl0.json", {1: 1.0, 2: 0.0}, seed=0),
                 _report(tmp_path / "rl1.json", {1: 0.0, 2: 0.0}, seed=1)]

    result = compare.compare(reference, candidate)

    deltas = {row["gt_id"]: row["delta"] for row in result["patients"]}
    assert deltas == {1: 0.0, 2: -0.5}
    assert result["paired"]["mean_reward_difference"] == -0.25
    assert (result["paired"]["wins"], result["paired"]["losses"], result["paired"]["ties"]) == (0, 1, 1)
    assert result["paired"]["candidate_replicates"] == 2


def test_the_same_report_cannot_count_twice_in_one_arm(tmp_path, dev_split):
    report = _report(tmp_path / "rl0.json", {1: 1.0, 2: 0.0}, seed=0)
    copy = tmp_path / "rl0-copy.json"
    copy.write_bytes(report.read_bytes())

    with pytest.raises(ValueError, match="Duplicate report"):
        compare.compare([report], [report, copy])


def test_incomplete_replicate_is_rejected(tmp_path, dev_split):
    complete = _report(tmp_path / "sft.json", {1: 0.5, 2: 0.5}, seed=0)
    partial = _report(tmp_path / "rl1.json", {1: 1.0}, seed=1)

    with pytest.raises(ValueError, match="exactly once"):
        compare.compare([complete], [complete.with_name("sft.json"), partial])


def test_heldout_is_never_a_comparison_split(tmp_path, dev_split):
    report = _report(tmp_path / "sft.json", {1: 0.5, 2: 0.5}, seed=0)

    with pytest.raises(ValueError, match="heldout is sealed"):
        compare.compare([report], [report], split="heldout")
