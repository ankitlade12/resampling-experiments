from imblearn.pipeline import Pipeline
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler


def train_model(estimator, params, X_train, y_train, refit=True):
    # train classifier with hyperparameter tuning
    # but without undersampling

    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_candidates="exhaust",
        factor=3,  # only a third of the candidates are promoted
        resource="n_estimators",  # the limiting resource
        max_resources=500,  # max number of trees
        min_resources=10,
        scoring="roc_auc",
        cv=3,
        random_state=10,
        refit=refit,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    return search


def train_model_with_undersampling(
    undersampler, estimator, scale, params, X_train, y_train, folds=3,
):
    # https://stackoverflow.com/questions/79748461/how-to-pass-pre-computed-folds-to-successivehalving-in-sklearn?

    if scale is True:
        pipe = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("sampler", undersampler),
                ("model", estimator),
            ]
        )
    else:
        pipe = Pipeline(
            [
                ("sampler", undersampler),
                ("model", estimator),
            ]
        )

    search = HalvingRandomSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_candidates="exhaust",
        factor=3,  # only a third of the candidates are promoted
        resource="model__n_estimators",  # the limiting resource
        max_resources=500,  # max number of trees
        min_resources=10,
        scoring="roc_auc",
        cv=folds,
        random_state=10,
        refit=True,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    return search
