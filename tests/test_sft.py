import json

import pytest

from training.sft import encode_demo, load_training_rows


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("The pinned tokenizer must already be cached; this test never downloads it")


@pytest.fixture
def demo():
    def call(name, arguments):
        return {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}
        }]}
    return {
        "gt_id": 7,
        "tools": json.dumps([{"type": "function", "function": {
            "name": "view_encounters", "parameters": {"type": "object", "properties": {
                "patient_id": {"type": "integer"}
            }}
        }}]),
        "messages": [
            {"role": "system", "content": "SYSTEM_CONTEXT_ONLY"},
            {"role": "user", "content": "USER_CONTEXT_ONLY"},
            call("view_encounters", {"patient_id": 7}),
            {"role": "tool", "content": "TOOL_OBSERVATION_ONLY"},
            call("submit_diagnosis", {"chronic_conditions": [{"icd10": "I10"}]}),
            {"role": "tool", "content": "SCORER_RESULT_ONLY"},
        ],
    }


def test_loss_targets_are_generated_tool_calls_and_eos_not_context_or_headers(tokenizer, demo):
    encoded = encode_demo(demo, tokenizer, max_length=8192)
    spans = []
    previous_masked = True
    for label in encoded["labels"]:
        if label != -100:
            if previous_masked:
                spans.append([])
            spans[-1].append(label)
        previous_masked = label == -100

    assert len(spans) == 2
    for span in spans:
        assert span[0] == tokenizer.convert_tokens_to_ids("<tool_call>")
        assert span[-1] == tokenizer.eos_token_id
        assert tokenizer.convert_tokens_to_ids("<|im_start|>") not in span
    targets = "".join(tokenizer.decode(span) for span in spans)
    assert "view_encounters" in targets and "submit_diagnosis" in targets and "I10" in targets
    assert "CONTEXT_ONLY" not in targets
    assert "OBSERVATION_ONLY" not in targets
    assert "SCORER_RESULT_ONLY" not in tokenizer.decode(encoded["input_ids"])


def test_overlong_demonstration_is_rejected_not_truncated(tokenizer, demo):
    with pytest.raises(ValueError):
        encode_demo(demo, tokenizer, max_length=16)


@pytest.mark.parametrize("ids", [[7, 9], [7, 99], [7, 7]])
def test_training_rejects_dev_unknown_or_duplicate_patients(tmp_path, ids):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps({"gt_id": i}) + "\n" for i in ids))
    with pytest.raises(ValueError):
        load_training_rows(path, train_ids={7, 8}, dev_ids={9})
