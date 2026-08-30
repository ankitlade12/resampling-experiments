import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from functions.evaluation import (
    METRIC_NAMES,
    evaluate_model_on_test_set,
    evaluate_predictions,
    select_f1_threshold,
)


@pytest.fixture
def fitted_model():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=10,
    )
    model = RandomForestClassifier(n_estimators=30, random_state=10)
    model.fit(X, y)
    model.decision_threshold_ = 0.4
    return model


@pytest.fixture
def test_data():
    return make_classification(
        n_samples=500,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=42,
    )


def test_returns_point_metrics_and_confidence_intervals(fitted_model, test_data):
    X, y = test_data
    result = evaluate_model_on_test_set(fitted_model, X, y, n_bootstrap=100)

    assert result["threshold"] == 0.4
    assert result["n_test"] == len(y)
    for metric in METRIC_NAMES:
        assert 0 <= result[metric] <= 1
        assert result[f"{metric}_ci_low"] <= result[metric]
        assert result[metric] <= result[f"{metric}_ci_high"]


def test_model_without_threshold_is_rejected(test_data):
    X, y = test_data
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="no decision_threshold"):
        evaluate_model_on_test_set(model, X, y, n_bootstrap=100)


def test_explicit_validation_threshold_is_accepted(test_data):
    X, y = test_data
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    result = evaluate_model_on_test_set(
        model, X, y, threshold=0.3, n_bootstrap=100
    )
    assert result["threshold"] == 0.3


def test_threshold_selection_uses_supplied_validation_predictions():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.4, 0.35, 0.9])
    threshold = select_f1_threshold(y, prob)
    assert threshold == pytest.approx(0.35)


def test_cluster_bootstrap_supported(test_data):
    _, y = test_data
    prob = np.where(y == 1, 0.7, 0.2)
    groups = np.repeat(np.arange(250), 2)
    result = evaluate_predictions(
        y,
        prob,
        0.5,
        groups=groups,
        n_bootstrap=100,
    )
    assert result["roc"] == pytest.approx(1.0)


def test_bootstrap_requires_enough_replicates(test_data):
    _, y = test_data
    prob = np.full(len(y), 0.5)
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_predictions(y, prob, 0.5, n_bootstrap=5)
