from common import metrics


def _rec(reward, w_recall, precision, n_pred=2, neutral=0, raw_precision=None, scored=True):
    if not scored:
        return {"gt_id": 0, "n_pred": 0, "reward": 0.0, "metrics": None}
    return {"gt_id": 0, "n_pred": n_pred, "reward": reward, "metrics": {
        "n_scored": 1, "problem_list_recall": w_recall, "weighted_problem_list_recall": w_recall,
        "problem_list_precision_neutral": precision,
        "problem_list_precision": precision if raw_precision is None else raw_precision,
        "n_neutral_predictions": neutral, "acuity_confusion_matrix": {"acute:acute": 1, "chronic:acute": 1}}}


def test_official_f1_is_harmonic_mean_of_means():
    out = metrics.aggregate([_rec(1.0, 1.0, 1.0), _rec(0.0, 0.0, 0.5)])
    assert out["weighted_recall"] == 0.5
    assert out["precision_neutral"] == 0.75
    assert abs(out["official_f1"] - 2 * 0.5 * 0.75 / 1.25) < 1e-9
    assert out["mean_reward"] == 0.5
    assert out["acuity_accuracy"] == 0.5


def test_unsubmitted_excluded_from_official_but_not_strict():
    out = metrics.aggregate([_rec(1.0, 1.0, 1.0), _rec(0, 0, 0, scored=False)])
    assert out["official_f1"] == 1.0
    assert abs(out["official_f1_strict"] - 0.5) < 1e-9
    assert out["mean_reward"] == 0.5
    assert out["valid_submit_rate"] == 0.5


def test_undefined_neutral_precision_is_excluded():
    # Every prediction unmatched and chart-neutral: the batch scorer drops it from the precision mean.
    undefined = _rec(0.0, 0.0, 0.0, n_pred=3, neutral=3, raw_precision=0.0)
    out = metrics.aggregate([_rec(1.0, 1.0, 1.0), undefined])
    assert out["precision_neutral"] == 1.0
    assert out["weighted_recall"] == 0.5


def test_count_predictions_tolerates_garbage():
    assert metrics.count_predictions(None) == 0
    assert metrics.count_predictions({"active_diagnoses": "I10"}) == 0
    assert metrics.count_predictions({"active_diagnoses": [{}], "chronic_conditions": [{}, {}]}) == 3
