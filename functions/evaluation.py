"""Leakage-safe model evaluation for imbalanced binary classification."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_NAMES = (
    "roc",
    "ap",
    "precision",
    "recall",
    "f1_score",
    "mcc",
    "ba",
    "brier",
    "gmean",
)


def select_f1_threshold(y, prob):
    """Select an F1-optimal threshold from validation or OOF predictions.

    This function is intentionally separate from test evaluation. Callers must
    learn the threshold from training-only validation predictions and freeze it
    before evaluating a held-out test partition.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype="float64")
    precisions, recalls, thresholds = precision_recall_curve(y, prob)

    if thresholds.size == 0:
        return 0.5

    # precision_recall_curve returns one more precision/recall value than
    # thresholds. The final point has no corresponding decision threshold.
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + np.finfo(float).eps
    )
    return float(thresholds[int(np.argmax(f1_scores))])


def predict_class(prob, threshold):
    """Convert positive-class probabilities using a pre-selected threshold."""
    if threshold is None:
        raise ValueError(
            "A validation-derived threshold is required; test labels must not be "
            "used to optimize a decision threshold."
        )
    return (np.asarray(prob) >= float(threshold)).astype(int)


def _geometric_mean(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return float(np.sqrt(sensitivity * specificity))


def calculate_classif_metrics(y, prob, threshold):
    """Return MCC, balanced accuracy, and G-mean at one frozen threshold."""
    pred = predict_class(prob, threshold)
    return (
        float(matthews_corrcoef(y, pred)),
        float(balanced_accuracy_score(y, pred)),
        _geometric_mean(y, pred),
    )


def _calculate_metrics(y, prob, threshold):
    pred = predict_class(prob, threshold)
    return {
        "roc": float(roc_auc_score(y, prob)),
        "ap": float(average_precision_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "ba": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, prob)),
        "gmean": _geometric_mean(y, pred),
    }


def _stratified_bootstrap_indices(y, rng):
    """Resample each class independently, preserving test prevalence."""
    parts = []
    for label in (0, 1):
        class_idx = np.flatnonzero(y == label)
        if class_idx.size == 0:
            raise ValueError("Both classes are required for bootstrap evaluation.")
        parts.append(rng.choice(class_idx, size=class_idx.size, replace=True))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return idx


def _cluster_members(groups):
    """Precompute row indices belonging to each patient/entity cluster."""
    groups = np.asarray(groups)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def _cluster_bootstrap_indices(members, rng):
    """Resample complete patient/entity clusters with replacement."""
    sampled_positions = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[position] for position in sampled_positions])


def bootstrap_metric_intervals(
    y,
    prob,
    threshold,
    *,
    groups=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=0,
):
    """Return percentile confidence intervals from test-set resampling.

    Ordinary datasets use a stratified row bootstrap. When ``groups`` is
    provided, complete clusters are resampled so repeated encounters from the
    same patient remain together.
    """
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100 for stable intervals.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype="float64")
    if groups is not None and len(groups) != len(y):
        raise ValueError("groups must have the same length as y.")

    rng = np.random.default_rng(random_state)
    samples = {metric: [] for metric in METRIC_NAMES}
    cluster_members = _cluster_members(groups) if groups is not None else None

    for _ in range(n_bootstrap):
        if groups is None:
            idx = _stratified_bootstrap_indices(y, rng)
        else:
            idx = _cluster_bootstrap_indices(cluster_members, rng)

        # A small cluster sample can occasionally contain one class only.
        if np.unique(y[idx]).size < 2:
            continue
        values = _calculate_metrics(y[idx], prob[idx], threshold)
        for metric, value in values.items():
            samples[metric].append(value)

    alpha = (1 - confidence_level) / 2
    intervals = {}
    for metric, values in samples.items():
        if not values:
            raise ValueError(f"No valid bootstrap replicates for {metric}.")
        low, high = np.quantile(values, [alpha, 1 - alpha])
        intervals[metric] = (float(low), float(high))
    return intervals


def paired_bootstrap_difference(
    y,
    reference_prob,
    candidate_prob,
    reference_threshold,
    candidate_threshold,
    *,
    metric="roc",
    groups=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=0,
):
    """Estimate a candidate-versus-reference effect with paired resampling.

    The same test rows or clusters are selected for both models in every
    replicate. Positive differences always favor the candidate; Brier loss is
    therefore ``reference - candidate`` while all other metrics use
    ``candidate - reference``.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}.")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100 for stable intervals.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    y = np.asarray(y).astype(int)
    reference_prob = np.asarray(reference_prob, dtype="float64")
    candidate_prob = np.asarray(candidate_prob, dtype="float64")
    if not len(y) == len(reference_prob) == len(candidate_prob):
        raise ValueError("y and both probability arrays must have equal length.")
    if groups is not None and len(groups) != len(y):
        raise ValueError("groups must have the same length as y.")

    def effect(indices):
        reference = _calculate_metrics(
            y[indices], reference_prob[indices], reference_threshold
        )[metric]
        candidate = _calculate_metrics(
            y[indices], candidate_prob[indices], candidate_threshold
        )[metric]
        return reference - candidate if metric == "brier" else candidate - reference

    point = effect(np.arange(len(y)))
    rng = np.random.default_rng(random_state)
    members = _cluster_members(groups) if groups is not None else None
    differences = []

    for _ in range(n_bootstrap):
        indices = (
            _stratified_bootstrap_indices(y, rng)
            if groups is None
            else _cluster_bootstrap_indices(members, rng)
        )
        if np.unique(y[indices]).size < 2:
            continue
        differences.append(effect(indices))

    if not differences:
        raise ValueError("No valid paired bootstrap replicates.")
    alpha = (1 - confidence_level) / 2
    low, high = np.quantile(differences, [alpha, 1 - alpha])
    return {
        "metric": metric,
        "difference": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "positive_favors": "candidate",
        "n_bootstrap_valid": len(differences),
    }


def evaluate_predictions(
    y,
    prob,
    threshold,
    *,
    groups=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=0,
):
    """Evaluate frozen predictions without making choices from test labels."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype="float64")
    point = _calculate_metrics(y, prob, threshold)
    intervals = bootstrap_metric_intervals(
        y,
        prob,
        threshold,
        groups=groups,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    result = {"threshold": float(threshold), "n_test": int(len(y))}
    for metric in METRIC_NAMES:
        result[metric] = point[metric]
        result[f"{metric}_ci_low"] = intervals[metric][0]
        result[f"{metric}_ci_high"] = intervals[metric][1]
    return result


def evaluate_model_on_test_set(
    model,
    X,
    y,
    *,
    threshold=None,
    groups=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=0,
):
    """Evaluate a fitted model using its training-derived frozen threshold."""
    if threshold is None:
        threshold = getattr(model, "decision_threshold_", None)
    if threshold is None:
        raise ValueError(
            "Model has no decision_threshold_. Retrain it with OOF threshold "
            "selection or pass a threshold learned from validation data."
        )

    prob = model.predict_proba(X)[:, 1]
    return evaluate_predictions(
        y,
        prob,
        threshold,
        groups=groups,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
