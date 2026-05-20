import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from functions.cv_undersamplers import undersample_data, train_model_w_undersampling

@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=10,
    )
    return X, y


@pytest.fixture
def undersampled_folds(imbalanced_data):
    X, y = imbalanced_data
    xtrainu, ytrainu, xtest, ytest, Xu, yu, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    return xtrainu, ytrainu, xtest, ytest, Xu, yu


@pytest.fixture
def params():
    return {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": range(2, 20),
        "min_samples_leaf": range(2, 20),
        "max_features": ["log2", "sqrt"],
    }


def test_returns_fitted_estimator(undersampled_folds, params):
    xtrainu, ytrainu, xtest, ytest, Xu, yu = undersampled_folds
    model = train_model_w_undersampling(
        RandomForestClassifier(random_state=10), params,
        xtrainu, ytrainu, xtest, ytest, Xu, yu
    )
    # a fitted estimator has estimators_ attribute
    assert hasattr(model, "estimators_")


def test_final_model_has_800_estimators(undersampled_folds, params):
    xtrainu, ytrainu, xtest, ytest, Xu, yu = undersampled_folds
    model = train_model_w_undersampling(
        RandomForestClassifier(random_state=10), params,
        xtrainu, ytrainu, xtest, ytest, Xu, yu
    )
    assert model.n_estimators == 800