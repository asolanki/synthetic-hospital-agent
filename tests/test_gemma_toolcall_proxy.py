"""The Gemma tool-call translator decides what counts as a tool call in the Gemma agent run, so its
acceptance rules are pinned here: known tools and literal arguments only, everything else stays text."""

from training.gemma_toolcall_proxy import extract_calls

TOOLS = [
    {"type": "function", "function": {"name": "open_chart", "parameters": {"properties": {"patient_id": {}}}}},
    {"type": "function", "function": {"name": "view_encounter_detail",
                                      "parameters": {"properties": {"encounter_id": {}, "verbose": {}}}}},
    {"type": "function", "function": {"name": "submit_diagnosis",
                                      "parameters": {"properties": {"active_diagnoses": {}, "chronic_conditions": {}}}}},
]


def test_tool_code_block_becomes_call_and_keeps_prose():
    calls, prose = extract_calls("Let me open the chart.\n```tool_code\n[open_chart(patient_id=1690)]\n```", TOOLS)
    assert calls == [{"name": "open_chart", "arguments": {"patient_id": 1690}}]
    assert prose == "Let me open the chart."


def test_bare_list_with_several_calls_and_positional_args():
    calls, _ = extract_calls("Next: [open_chart(7), view_encounter_detail(6802, verbose=true)]", TOOLS)
    assert calls == [{"name": "open_chart", "arguments": {"patient_id": 7}},
                     {"name": "view_encounter_detail", "arguments": {"encounter_id": 6802, "verbose": True}}]


def test_submission_with_json_literals_parses():
    text = ('```tool_code\n[submit_diagnosis(active_diagnoses=[{"icd10": "I21.01", "name": "STEMI", '
            '"acuity": "acute"}], chronic_conditions=null)]\n```')
    calls, _ = extract_calls(text, TOOLS)
    assert calls[0]["arguments"] == {"active_diagnoses": [{"icd10": "I21.01", "name": "STEMI", "acuity": "acute"}],
                                     "chronic_conditions": None}


def test_unknown_tool_or_non_literal_argument_stays_text():
    assert extract_calls("```tool_code\n[view_results(patient_id=3)]\n```", TOOLS) is None
    assert extract_calls("```tool_code\n[open_chart(patient_id=pid)]\n```", TOOLS) is None
    assert extract_calls("I will open_chart(patient_id=3) next.", TOOLS) is None


def test_json_block_is_ignored_but_tool_code_block_is_used():
    text = '```json\n{"note": 1}\n```\nNow:\n```tool_code\n[open_chart(patient_id=5)]\n```'
    calls, _ = extract_calls(text, TOOLS)
    assert calls == [{"name": "open_chart", "arguments": {"patient_id": 5}}]
