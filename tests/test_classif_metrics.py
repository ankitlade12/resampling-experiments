import numpy as np
import pytest
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from functions.evaluation import calculate_classif_metrics


def _brute_force(y, prob):
    """Reference implementation: scan every unique probability threshold."""
    thresholds = np.unique(prob)
    mcc = [matthews_corrcoef(y, prob >= t) for t in thresholds]
    ba = [balanced_accuracy_score(y, prob >= t) for t in thresholds]
    gmean = [geometric_mean_score(y, prob >= t) for t in thresholds]
    return max(mcc), max(ba), max(gmean)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_brute_force_loop(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(2000) < 0.15).astype(int)
    prob = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.3, 2000), 0, 1)

    fast = calculate_classif_metrics(y, prob)
    slow = _brute_force(y, prob)

    assert fast == pytest.approx(slow, abs=1e-9)


def test_handles_tied_probabilities():
    # Discrete probabilities (e.g. from a small random forest) produce many ties.
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    prob = np.array([0.2, 0.8, 0.2, 0.6, 0.8, 0.4, 0.2, 0.6])
    assert calculate_classif_metrics(y, prob) == pytest.approx(
        _brute_force(y, prob), abs=1e-9
    )


def test_returns_three_floats():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(int)
    prob = rng.random(200)
    result = calculate_classif_metrics(y, prob)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)
