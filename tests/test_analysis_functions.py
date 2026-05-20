import pytest
import numpy as np
from functions.analysis import create_df


@pytest.fixture
def sample_scores_dict():
    metrics = ["roc", "roc_std", "ap", "ap_std", "precision", "precision_std",
               "recall", "recall_std", "f1_score", "f1_std", "mcc", "mcc_std",
               "ba", "ba_std", "thresh", "tresh_std", "brier", "brier_std",
               "gmean", "gmean_std"]
    return {
        "dataset1": {
            "logit": dict(zip(metrics, np.random.rand(len(metrics)))),
            "knn":   dict(zip(metrics, np.random.rand(len(metrics)))),
            "tree":  dict(zip(metrics, np.random.rand(len(metrics)))),
        }
    }


def test_create_df_shape(sample_scores_dict):
    df = create_df(sample_scores_dict, "dataset1", ["logit", "knn"])
    assert df.shape == (2, 20), f"Expected shape (2, 20), got {df.shape}"


def test_create_df_columns(sample_scores_dict):
    df = create_df(sample_scores_dict, "dataset1", ["logit"])
    expected_cols = ["roc", "roc_std", "ap", "ap_std", "precision", "precision_std",
                     "recall", "recall_std", "f1_score", "f1_std", "mcc", "mcc_std",
                     "ba", "ba_std", "brier", "brier_std", "gmean", "gmean_std",
                     "thresh", "tresh_std"]
    assert list(df.columns) == expected_cols


def test_create_df_models_as_index(sample_scores_dict):
    models = ["logit", "knn"]
    df = create_df(sample_scores_dict, "dataset1", models)
    assert list(df.index) == models, f"Expected index {models}, got {list(df.index)}"


def test_create_df_fillna(sample_scores_dict):
    # Introduce a NaN manually
    sample_scores_dict["dataset1"]["logit"]["roc"] = None
    df = create_df(sample_scores_dict, "dataset1", ["logit"])
    assert df.isnull().sum().sum() == 0, f"Expected no NaN values, got {df.isnull().sum().sum()} NaNs"


def test_create_df_missing_model(sample_scores_dict):
    with pytest.raises(KeyError, match="nonexistent_model"):
        create_df(sample_scores_dict, "dataset1", ["nonexistent_model"])