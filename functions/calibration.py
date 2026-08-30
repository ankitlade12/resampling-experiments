"""Training-only probability calibration for fitted binary classifiers."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def _logit(probability):
    eps = np.finfo("float64").eps
    probability = np.clip(np.asarray(probability, dtype="float64"), eps, 1 - eps)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def fit_sigmoid_calibrator(probability, y):
    """Fit Platt-style sigmoid calibration on original-prevalence OOF data."""
    probability = np.asarray(probability, dtype="float64")
    y = np.asarray(y).astype(int)
    if len(probability) != len(y):
        raise ValueError("probability and y must have equal length.")
    if np.unique(y).size != 2:
        raise ValueError("Both classes are required for probability calibration.")
    calibrator = LogisticRegression(random_state=10)
    calibrator.fit(_logit(probability), y)
    return calibrator


def calibrate_probability(calibrator, probability):
    """Transform raw positive-class scores with a fitted sigmoid."""
    return calibrator.predict_proba(_logit(probability))[:, 1]


class ProbabilityCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """A fitted classifier whose probabilities pass through an OOF calibrator."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    @property
    def classes_(self):
        return self.base_model.classes_

    @property
    def estimators_(self):
        return self.base_model.estimators_

    @property
    def n_estimators(self):
        return self.base_model.n_estimators

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        positive = calibrate_probability(self.calibrator, raw)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
