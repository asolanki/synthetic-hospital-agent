from training.recode import recode

TABLE = {"autoimmune pericarditis": "M32.12", "dehydration": "E86.0"}


def test_only_table_names_are_recoded_and_same_category_is_not_counted():
    submission = {
        "active_diagnoses": [
            {"icd10": "I30.9", "name": "  Autoimmune Pericarditis ", "acuity": "acute"},  # wrong code -> table code
            {"icd10": "E86.1", "name": "Dehydration", "acuity": "acute"},  # same E86 category: unchanged count
            {"icd10": "R50.9", "name": "Fever of unknown origin", "acuity": "acute"},  # not in table: kept
        ],
        "chronic_conditions": ["not an entry"],
    }

    out, entries, changed = recode(submission, TABLE)

    assert [e["icd10"] for e in out["active_diagnoses"]] == ["M32.12", "E86.1", "R50.9"]
    assert (entries, changed) == (3, 1)
    assert submission["active_diagnoses"][0]["icd10"] == "I30.9"


def test_missing_answer_stays_missing():
    assert recode(None, TABLE) == (None, 0, 0)
