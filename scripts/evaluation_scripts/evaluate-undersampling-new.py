"""
Evaluate the undersampling models trained on the new datasets.

Loads each model saved by train-undersampling-new.py, evaluates it on the test
set at the training-derived frozen threshold with bootstrap confidence
intervals, and stores a
merged results pickle in models/undersampling-new/, keyed by dataset then by
``{estimator}_{undersampler}`` (matching evaluate-undersampling.py).

The dataset/undersampler scope mirrors the training script: the full suite on
htru2/default_credit/secom and the row-reducing methods on diabetes130/creditcard.
"""

import pickle
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from functions.evaluation import evaluate_model_on_test_set
from functions.hard_data import load_hard_dataset
from functions.imbalanced_data import load_imbalanced_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

FULL_SUITE = [
    "rus",
    "cnn",
    "tomek",
    "oss",
    "enn",
    "renn",
    "allknn",
    "ncr",
    "nm1",
    "nm2",
]
ROW_REDUCERS = ["rus", "nm1", "nm2"]

DATASETS = [
    ("htru2", load_imbalanced_dataset, FULL_SUITE),
    ("default_credit", load_hard_dataset, FULL_SUITE),
    ("diabetes130", load_hard_dataset, ROW_REDUCERS),
    ("creditcard", load_imbalanced_dataset, ROW_REDUCERS),
    ("secom", load_hard_dataset, FULL_SUITE),
]

MODELS_DIR = REPO_ROOT / "models" / "undersampling-new"

scores_dict = {}
predictions_dict = {}

for dataset, loader, undersamplers in tqdm(DATASETS, desc="Datasets"):
    if loader is load_hard_dataset:
        _, X_test, _, y_test, metadata = loader(dataset, return_metadata=True)
        groups = metadata["test_groups"]
    else:
        _, X_test, _, y_test = loader(dataset)
        groups = None

    scores_dict[dataset] = {}
    predictions_dict[dataset] = {
        "y": np.asarray(y_test),
        "groups": None if groups is None else np.asarray(groups),
        "models": {},
    }
    for undersampler in undersamplers:
        for estimator in estimator_dict:
            model = joblib.load(
                MODELS_DIR / f"{dataset}_{estimator}_{undersampler}.pkl"
            )
            scores_dict[dataset][f"{estimator}_{undersampler}"] = (
                evaluate_model_on_test_set(
                    model,
                    X_test,
                    y_test,
                    groups=groups,
                )
            )
            predictions_dict[dataset]["models"][f"{estimator}_{undersampler}"] = {
                "probability": model.predict_proba(X_test)[:, 1],
                "threshold": model.decision_threshold_,
            }

with open(MODELS_DIR / "results", "wb") as fp:
    pickle.dump(scores_dict, fp)

with open(MODELS_DIR / "predictions", "wb") as fp:
    pickle.dump(predictions_dict, fp)
