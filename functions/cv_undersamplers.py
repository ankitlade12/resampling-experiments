"""
Cross-validation utilities for training models on pre-undersampled data.

The key design choice here is to undersample once before hyperparameter search
rather than embedding the undersampler inside a pipeline. This avoids re-running
slow undersamplers (e.g., CNN) on every candidate and resource level during
successive halving, at the cost of some statistical purity.

https://stackoverflow.com/questions/79748461/how-to-pass-pre-computed-folds-to-successivehalving-in-sklearn?
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from configs.experiment import (
    CV_RANDOM_STATE,
    CV_SPLITS,
    HALVING_CANDIDATES,
    HALVING_FACTOR,
    HALVING_RESOURCES,
    PARAMETER_RANDOM_STATE,
    THRESHOLD_METRIC,
)
from functions.calibration import (
    ProbabilityCalibratedClassifier,
    calibrate_probability,
    fit_sigmoid_calibrator,
)
from functions.evaluation import select_f1_threshold


def undersample_data(undersampler, X, y, scale):
    """
    Apply an undersampler to a 3-fold stratified split and return the results.

    Undersampling is applied to each training fold independently, so the
    test folds always reflect the original class distribution. A final
    undersampled version of the full training set is also returned for
    refitting the best model after hyperparameter search.

    Parameters
    ----------
    undersampler : imblearn sampler
        A fitted/unfitted imblearn undersampler with a `fit_resample` method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    scale : bool, default=False
        If True, apply MinMaxScaler before undersampling so that distance-based
        undersamplers (e.g., CNN) operate on comparable feature ranges. The
        returned data is always in the original scale.

    Returns
    -------
    If return_index=False (default):
        xtrainu : list of 3 arrays
            Undersampled feature matrices for each training fold.
        ytrainu : list of 3 arrays
            Undersampled target vectors for each training fold.
        xtest : list of 3 arrays
            Original (non-undersampled) feature matrices for each test fold.
        ytest : list of 3 arrays
            Original target vectors for each test fold.
        Xu : array
            Undersampled feature matrix for the full training set.
        yu : array
            Undersampled target vector for the full training set.
        stats : dict
            Summary of undersampling effect on the full training set:
            original_size, undersampled_size, removed, removed_pct.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    skf = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    xtrain, ytrain, xtest, ytest = [], [], [], []

    for train_index, test_index in skf.split(X, y):
        xtrain.append(X[train_index])
        ytrain.append(y[train_index])
        xtest.append(X[test_index])
        ytest.append(y[test_index])

    xtrainu, ytrainu = [], []

    for data, target in zip(xtrain, ytrain):

        if scale is True:
            # Scale only to guide the undersampler (e.g., distance-based
            # methods like CNN need consistent feature ranges), but return
            # the original-scale data so models are trained on raw values.
            undersampler.fit_resample(MinMaxScaler().fit_transform(data), target)
            datau = data[undersampler.sample_indices_]
            targetu = target[undersampler.sample_indices_]
        else:
            datau, targetu = undersampler.fit_resample(data, target)

        xtrainu.append(datau)
        ytrainu.append(targetu)

    # also return undersampled train set for training final model
    if scale is True:
        undersampler.fit_resample(MinMaxScaler().fit_transform(X), y)
        Xu = X[undersampler.sample_indices_]
        yu = y[undersampler.sample_indices_]
    else:
        Xu, yu = undersampler.fit_resample(X, y)

    stats = {
        "original_size": len(X),
        "undersampled_size": len(Xu),
        "removed": len(X) - len(Xu),
        "removed_pct": round((1 - len(Xu) / len(X)) * 100, 1),
    }

    return xtrainu, ytrainu, xtest, ytest, Xu, yu, stats


def train_model_w_undersampling(
    model, params, xtrainu, ytrainu, xtest, ytest, Xu, yu, scoring="roc_auc"
):
    """
    Tune and train a model on pre-undersampled data.

    Undersampling is applied once upfront (via `undersample_data`) rather than
    inside a pipeline, so slow undersamplers like CNN are not re-run on every
    candidate during hyperparameter search.
    The tuning uses the same successive-halving budget as ``train_model``:
    100/34/12/4/2 candidates at 10/30/90/270/810 estimators, respectively.
    The winner is refit with 810 estimators on all undersampled training data.

    Parameters
    ----------
    model : estimator
        An unfitted sklearn-compatible classifier with `n_estimators`.
    params : dict
        Hyperparameter search space passed to ParameterSampler.
    xtrainu, ytrainu : list of arrays
        Per-fold undersampled training sets produced by `undersample_data`.
    xtest, ytest : list of arrays
        Per-fold held-out test sets produced by `undersample_data`.
    Xu, yu : arrays
        Full undersampled training set for final refit.
    scoring : str, default="roc_auc"
        Scoring metric to use for hyperparameter tuning.
        One of "log_loss", "roc_auc", or "brier".
    """

    valid_scorings = ("log_loss", "roc_auc", "brier")
    if scoring not in valid_scorings:
        raise ValueError(f"scoring must be one of {valid_scorings}, got '{scoring}'")

    if not (len(xtrainu) == len(ytrainu) == len(xtest) == len(ytest) == CV_SPLITS):
        raise ValueError(f"Expected {CV_SPLITS} precomputed folds.")

    candidates = list(
        ParameterSampler(
            params,
            n_iter=HALVING_CANDIDATES[0],
            random_state=PARAMETER_RANDOM_STATE,
        )
    )
    best_y_oof = None
    best_prob_oof = None

    # Explicitly mirror sklearn's successive-halving schedule because every
    # candidate must be trained on a different pre-undersampled fold.
    for round_index, (resource, keep) in enumerate(
        zip(HALVING_RESOURCES, HALVING_CANDIDATES[1:] + (1,))
    ):
        results = []
        for params_ in candidates:
            fold_scores = []
            fold_truth = []
            fold_probabilities = []
            temp_params = {**params_, "n_estimators": resource}

            for i in range(CV_SPLITS):
                clf = clone(model)
                clf.set_params(**temp_params)
                clf.fit(xtrainu[i], ytrainu[i])
                y_pred = clf.predict_proba(xtest[i])[:, 1]
                fold_truth.append(np.asarray(ytest[i]))
                fold_probabilities.append(y_pred)

                if scoring == "log_loss":
                    fold_scores.append(log_loss(ytest[i], y_pred))
                elif scoring == "roc_auc":
                    fold_scores.append(roc_auc_score(ytest[i], y_pred))
                else:
                    fold_scores.append(brier_score_loss(ytest[i], y_pred))

            results.append(
                (
                    params_,
                    float(np.mean(fold_scores)),
                    np.concatenate(fold_truth),
                    np.concatenate(fold_probabilities),
                )
            )

        ranked = sorted(
            results,
            key=lambda item: item[1],
            reverse=scoring == "roc_auc",
        )
        winners = ranked[:keep]
        candidates = [item[0] for item in winners]

        if round_index == len(HALVING_RESOURCES) - 1:
            best_params, best_score, best_y_oof, best_prob_oof = winners[0]

    best_params_final = best_params.copy()
    best_params_final["n_estimators"] = HALVING_RESOURCES[-1]

    model = clone(model)
    model.set_params(**best_params_final)
    model.fit(Xu, yu)
    calibrator = fit_sigmoid_calibrator(best_prob_oof, best_y_oof)
    calibrated_oof = calibrate_probability(calibrator, best_prob_oof)
    model = ProbabilityCalibratedClassifier(model, calibrator)
    model.decision_threshold_ = select_f1_threshold(best_y_oof, calibrated_oof)
    model.threshold_selection_ = {
        "metric": THRESHOLD_METRIC,
        "source": f"{CV_SPLITS}-fold out-of-fold training predictions",
        "random_state": CV_RANDOM_STATE,
    }
    model.search_protocol_ = {
        "factor": HALVING_FACTOR,
        "candidate_schedule": HALVING_CANDIDATES,
        "resource_schedule": HALVING_RESOURCES,
        "scoring": scoring,
        "best_score": best_score,
    }
    model.probability_calibration_ = {
        "method": "sigmoid_on_logit",
        "source": f"{CV_SPLITS}-fold out-of-fold training predictions",
        "original_prevalence": float(np.asarray(best_y_oof).mean()),
        "refit_prevalence": float(np.asarray(yu).mean()),
    }

    return model
