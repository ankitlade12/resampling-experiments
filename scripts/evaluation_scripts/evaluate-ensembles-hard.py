"""
Evaluate pre-trained ensemble classifiers on the *hard* imbalanced datasets.

Loads each model saved by train-ensembles-hard.py, evaluates it on the test set
using bootstrapped samples (metrics at their optimal threshold), and stores the
results as a pickle in the model folder.
"""

import pickle
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from functions.evaluation import evaluate_model_on_test_set
from functions.hard_data import DATASETS_HARD, load_hard_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

MODELS_DIR = REPO_ROOT / "models" / "ensembles-hard"

scores_dict = {}

for dataset in tqdm(DATASETS_HARD, desc="Datasets"):
    _, X_test, _, y_test = load_hard_dataset(dataset)

    scores_dict[dataset] = {}

    for estimator in tqdm(estimator_dict, desc=dataset, leave=False):
        search = joblib.load(MODELS_DIR / f"{dataset}_{estimator}.pkl")
        scores_dict[dataset][estimator] = evaluate_model_on_test_set(
            search, X_test, y_test
        )

with open(MODELS_DIR / "results", "wb") as fp:
    pickle.dump(scores_dict, fp)
