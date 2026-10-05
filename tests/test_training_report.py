import asyncio
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from training import report


@pytest.mark.parametrize("episode_ok", [True, False])
def test_model_call_errors_remain_visible_even_when_episode_is_ok(tmp_path, monkeypatch, episode_ok):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [7, 8]}))
    monkeypatch.setattr(report.config, "SPLIT_FILE", split)
    instances = [SimpleNamespace(gt_id=i, gt_diagnoses=[], encounters=[
        {"encounter_order": 0, "encounter_id": i}
    ]) for i in (7, 8)]
    monkeypatch.setattr(report.benchmark, "load_instances", lambda _: instances)
    monkeypatch.setattr(report, "Simulator", lambda: nullcontext())
    episodes = []
    for gt_id in (7, 8):
        ok = episode_ok if gt_id == 7 else True
        calls = [{"model": "test-model", "error": {
            "type": "ProviderError", "status_code": 400,
            "message": "Maximum context length exceeded",
        }}] if gt_id == 7 else []
        episodes.append({
            "ok": ok, "errors": [], "task": {"data": {"gt_id": gt_id, "budget": 40}},
            "num_input_tokens": 0, "num_total_tokens": 0,
            "traces": [{"metrics": {}, "nodes": [], "calls": calls, "errors": [],
                        "rewards": {"official_score": {"score": 0}} if ok else {}}],
        })
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in episodes))

    result = asyncio.run(report.report(path, "dev"))

    summary = result["summary"]
    assert summary["n"] == 2
    assert summary["mean_reward"] == 0
    assert summary["rollout_error_rate"] == (0 if episode_ok else 0.5)
    assert summary["model_call_error_episode_rate"] == 0.5
    assert summary["model_call_error_types"] == {"ProviderError": 1}
    assert summary["mean_actions"] is None
