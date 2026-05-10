from imblearn.pipeline import Pipeline
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler


def train_model(estimator, params, X_train, y_train, refit=True):
    """
    Train classifier with hyperparameter tuning
    but without undersampling.
    """

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
