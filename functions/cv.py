import numpy as np
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, StratifiedKFold

from configs.experiment import (
    CV_RANDOM_STATE,
    CV_SPLITS,
    HALVING_FACTOR,
    HALVING_MAX_RESOURCES,
    HALVING_MIN_RESOURCES,
    PARAMETER_RANDOM_STATE,
    THRESHOLD_METRIC,
)
from functions.evaluation import select_f1_threshold


def _subset(data, indices):
    """Index pandas or numpy inputs without discarding feature metadata."""
    return data.iloc[indices] if hasattr(data, "iloc") else data[indices]


def _oof_threshold(estimator, X, y, sample_weight=None):
    """Learn a decision threshold from training-only OOF scores."""
    y_array = np.asarray(y)
    oof_prob = np.empty(len(y_array), dtype="float64")
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    for train_idx, valid_idx in cv.split(X, y_array):
        fold_model = clone(estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight)[train_idx]
        fold_model.fit(
            _subset(X, train_idx),
            y_array[train_idx],
            **fit_kwargs,
        )
        oof_prob[valid_idx] = fold_model.predict_proba(_subset(X, valid_idx))[:, 1]

    return select_f1_threshold(y_array, oof_prob)


def _attach_oof_threshold(container, estimator, X, y, sample_weight=None):
    container.decision_threshold_ = _oof_threshold(
        estimator, X, y, sample_weight=sample_weight
    )
    container.threshold_selection_ = {
        "metric": THRESHOLD_METRIC,
        "source": f"{CV_SPLITS}-fold out-of-fold training predictions",
        "random_state": CV_RANDOM_STATE,
    }


def get_sample_weights(y_train):
    """
    Compute sample weight arrays for cost-sensitive learning.

    Returns a dict mapping integer weight -> sample weight array, where the
    minority class (label 1) receives the weight and the majority class gets 1.
    Candidates are IR, IR//2, and IR//3 (integers >= 2, duplicates dropped).
    """
    IR = int(round((y_train == 0).sum() / (y_train == 1).sum()))
    candidates = dict.fromkeys(w for w in (IR, IR // 2, IR // 3) if w >= 2)
    return {w: np.where(y_train == 1, w, 1).astype(int) for w in candidates}


def train_model(
    estimator,
    params,
    X_train,
    y_train,
    scoring="roc_auc",
    refit=True,
    n_jobs=-1,
    sample_weight=None,
):
    """
    Train classifier with hyperparameter tuning
    using successive halving and without undersampling.

    ``n_jobs`` controls the parallelism of the search itself (defaults to -1, all
    cores). Set ``n_jobs=1`` to run the candidate search sequentially; this avoids
    a nested-parallelism deadlock that can occur on macOS when the search workers
    and a parallel estimator both spawn joblib/loky processes. It does not change
    the fitted models, only how they are scheduled.
    """

    # CatBoostClassifier does not recognize n_estimators as hyperparameter.
    # https://github.com/scikit-learn/scikit-learn/issues/19844

    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_candidates="exhaust",  # the number of candidates to evaluate at the first iteration
        factor=HALVING_FACTOR,
        resource="n_estimators",  # the limiting resource
        max_resources=HALVING_MAX_RESOURCES,
        min_resources=HALVING_MIN_RESOURCES,
        scoring=scoring,
        cv=StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=CV_RANDOM_STATE,
        ),
        random_state=PARAMETER_RANDOM_STATE,
        refit=refit,
        n_jobs=n_jobs,
    )

    # Only forward sample_weight when set: some estimators (e.g. imbalanced-learn's
    # EasyEnsembleClassifier) do not accept a sample_weight argument at all, so
    # passing sample_weight=None would raise instead of being a harmless no-op.
    if sample_weight is not None:
        search.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        search.fit(X_train, y_train)
    _attach_oof_threshold(
        search,
        search.best_estimator_,
        X_train,
        y_train,
        sample_weight=sample_weight,
    )
    return search


def train_basic_model(
    estimator, params, X_train, y_train, scoring="roc_auc", refit=True
):
    """
    Train classifier with hyperparameter tuning
    using randomized search and without undersampling.
    """

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=20,
        scoring=scoring,
        cv=StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=CV_RANDOM_STATE,
        ),
        random_state=PARAMETER_RANDOM_STATE,
        refit=refit,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    _attach_oof_threshold(search, search.best_estimator_, X_train, y_train)
    return search
