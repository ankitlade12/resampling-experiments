import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from functions.calibration import (
    ProbabilityCalibratedClassifier,
    calibrate_probability,
    fit_sigmoid_calibrator,
)


def test_sigmoid_calibration_is_bounded_and_monotonic():
    raw = np.linspace(0, 1, 100)
    y = (raw > 0.7).astype(int)
    calibrator = fit_sigmoid_calibrator(raw, y)
    calibrated = calibrate_probability(calibrator, raw)

    assert np.all((0 <= calibrated) & (calibrated <= 1))
    assert np.all(np.diff(calibrated) >= 0)


def test_calibrated_classifier_returns_valid_probabilities():
    X, y = make_classification(n_samples=200, random_state=10)
    model = RandomForestClassifier(n_estimators=5, random_state=10).fit(X, y)
    raw = model.predict_proba(X)[:, 1]
    wrapped = ProbabilityCalibratedClassifier(model, fit_sigmoid_calibrator(raw, y))

    probability = wrapped.predict_proba(X)
    assert probability.shape == (len(y), 2)
    assert np.allclose(probability.sum(axis=1), 1)
    assert wrapped.n_estimators == 5
    assert hasattr(wrapped, "estimators_")
