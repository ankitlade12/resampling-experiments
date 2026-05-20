from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV


def train_model(estimator, params, X_train, y_train, scoring="roc_auc", refit=True):
    """
    Train classifier with hyperparameter tuning
    using successive halving and without undersampling.
    """

    # CatBoostClassifier does not recognize n_estimators as hyperparameter.
    # https://github.com/scikit-learn/scikit-learn/issues/19844

    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_candidates="exhaust",  # the number of candidates to evaluate at the first iteration
        factor=3,  # only a third of the candidates are promoted
        resource="n_estimators",  # the limiting resource
        max_resources=1000,  # max number of trees (or samples)
        min_resources=10,  # min number of trees (or samples)
        scoring=scoring,  # proper scoring function (ensures probabilistic distribution)
        cv=3,  # uses StratifiedKFold by default
        random_state=10,
        refit=refit,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
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
        cv=3,
        random_state=10,
        refit=refit,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    return search
