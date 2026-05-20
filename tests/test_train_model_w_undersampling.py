import pytest
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from functions.cv_undersamplers import train_model_w_undersampling, undersample_data


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
        RandomForestClassifier(random_state=10),
        params,
        xtrainu,
        ytrainu,
        xtest,
        ytest,
        Xu,
        yu,
    )
    # a fitted estimator has estimators_ attribute
    assert hasattr(model, "estimators_")


def test_final_model_has_800_estimators(undersampled_folds, params):
    xtrainu, ytrainu, xtest, ytest, Xu, yu = undersampled_folds
    model = train_model_w_undersampling(
        RandomForestClassifier(random_state=10),
        params,
        xtrainu,
        ytrainu,
        xtest,
        ytest,
        Xu,
        yu,
    )
    assert model.n_estimators == 800


def test_scoring_metrics_produce_different_results(undersampled_folds, params):
    xtrainu, ytrainu, xtest, ytest, Xu, yu = undersampled_folds

    model_log_loss = train_model_w_undersampling(
        RandomForestClassifier(random_state=10),
        params,
        xtrainu,
        ytrainu,
        xtest,
        ytest,
        Xu,
        yu,
        scoring="log_loss",
    )

    model_roc_auc = train_model_w_undersampling(
        RandomForestClassifier(random_state=10),
        params,
        xtrainu,
        ytrainu,
        xtest,
        ytest,
        Xu,
        yu,
        scoring="roc_auc",
    )

    model_brier = train_model_w_undersampling(
        RandomForestClassifier(random_state=10),
        params,
        xtrainu,
        ytrainu,
        xtest,
        ytest,
        Xu,
        yu,
        scoring="brier",
    )

    # extract the hyperparameters selected by each scoring metric
    params_log_loss = {k: v for k, v in model_log_loss.get_params().items()}
    params_roc_auc = {k: v for k, v in model_roc_auc.get_params().items()}
    params_brier = {k: v for k, v in model_brier.get_params().items()}

    # different scoring metrics should select different hyperparameter configurations
    assert params_log_loss != params_roc_auc or params_log_loss != params_brier


def test_invalid_scoring_raises(undersampled_folds, params):
    xtrainu, ytrainu, xtest, ytest, Xu, yu = undersampled_folds
    with pytest.raises(ValueError, match="scoring must be one of"):
        train_model_w_undersampling(
            RandomForestClassifier(random_state=10),
            params,
            xtrainu,
            ytrainu,
            xtest,
            ytest,
            Xu,
            yu,
            scoring="invalid_metric",
        )
