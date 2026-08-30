"""
Evaluate pre-trained ensemble classifiers on the strong-signal, severely imbalanced datasets.

Loads each model saved by train-ensembles-imbalanced.py and evaluates it at its
training-derived frozen threshold with bootstrap confidence intervals. Test
predictions are retained for paired model comparisons.
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
from functions.imbalanced_data import DATASETS_IMBALANCED, load_imbalanced_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

MODELS_DIR = REPO_ROOT / "models" / "ensembles-imbalanced"

scores_dict = {}
predictions_dict = {}

for dataset in tqdm(DATASETS_IMBALANCED, desc="Datasets"):
    _, X_test, _, y_test = load_imbalanced_dataset(dataset)

    scores_dict[dataset] = {}
    predictions_dict[dataset] = {
        "y": np.asarray(y_test),
        "groups": None,
        "models": {},
    }

    for estimator in tqdm(estimator_dict, desc=dataset, leave=False):
        search = joblib.load(MODELS_DIR / f"{dataset}_{estimator}.pkl")
        scores_dict[dataset][estimator] = evaluate_model_on_test_set(
            search, X_test, y_test
        )
        predictions_dict[dataset]["models"][estimator] = {
            "probability": search.predict_proba(X_test)[:, 1],
            "threshold": search.decision_threshold_,
        }

with open(MODELS_DIR / "results", "wb") as fp:
    pickle.dump(scores_dict, fp)

with open(MODELS_DIR / "predictions", "wb") as fp:
    pickle.dump(predictions_dict, fp)
