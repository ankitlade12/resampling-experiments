import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from functions.cv import train_model


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=10)
    return X, y


@pytest.fixture
def fitted_search(classification_data):
    X_train, y_train = classification_data
    estimator = RandomForestClassifier(random_state=10)
    params = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": range(2, 20),
        "min_samples_leaf": range(2, 20),
        "max_features": ["log2", "sqrt"],
    }
    return train_model(estimator, params, X_train, y_train)


def test_n_resources(fitted_search):
    # the number of n_estimators used at each iteration
    assert list(fitted_search.n_resources_) == [10, 30, 90, 270, 810]


def test_n_iterations(fitted_search):
    assert fitted_search.n_iterations_ == 5


def test_n_candidates(fitted_search):
    # the number of hyperparameter combinations tested at each iteration
    assert list(fitted_search.n_candidates_) == [100, 34, 12, 4, 2]
