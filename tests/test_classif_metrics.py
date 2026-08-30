import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from functions.evaluation import calculate_classif_metrics, predict_class


def test_metrics_use_one_frozen_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.4, 0.7, 0.3, 0.6, 0.9])
    threshold = 0.5
    pred = prob >= threshold

    mcc, ba, gmean = calculate_classif_metrics(y, prob, threshold)
    sensitivity = 2 / 3
    specificity = 2 / 3

    assert mcc == pytest.approx(matthews_corrcoef(y, pred))
    assert ba == pytest.approx(balanced_accuracy_score(y, pred))
    assert gmean == pytest.approx(np.sqrt(sensitivity * specificity))


def test_threshold_changes_all_threshold_dependent_metrics_together():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.4, 0.7, 0.3, 0.6, 0.9])
    assert calculate_classif_metrics(y, prob, 0.5) != pytest.approx(
        calculate_classif_metrics(y, prob, 0.85)
    )


def test_predict_class_requires_preselected_threshold():
    with pytest.raises(ValueError, match="validation-derived"):
        predict_class(np.array([0.2, 0.8]), None)


def test_predict_class_applies_threshold():
    assert predict_class(np.array([0.2, 0.5, 0.8]), 0.5).tolist() == [0, 1, 1]
