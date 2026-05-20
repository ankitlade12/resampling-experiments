"""
Train classical machine learning classifiers on various imbalanced datasets.

Models trained: logistic regression, knn, svc and decision trees from scikit-learn.
Hyperparameter tuning uses randomized seach limiting the tests to 20 because these
models have few hyperparameters.
"""
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from tqdm import tqdm

import numpy as np

from functions.data import DATASETS_LS, load_dataset
from configs.basic_models import estimator_dict
from configs.hyperparams import hyperparam_basic_dict
from functions.cv import train_basic_model

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)

OUTPUT_DIR = REPO_ROOT / "models" / "other"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

problematic = []
for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    for estimator, params in tqdm(
        zip(estimator_dict, hyperparam_basic_dict),
        desc=dataset,
        total=len(estimator_dict),
        leave=False,
    ):
        search = train_basic_model(
            estimator_dict[estimator],
            hyperparam_basic_dict[params],
            X_train,
            y_train,
        )
        joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}.pkl")

        n_unique = len(np.unique(search.best_estimator_.predict_proba(X_train)[:, 1]))
        if n_unique < 10:
            problematic.append({
                "dataset": dataset,
                "estimator": estimator,
                "n_unique": n_unique,
            })

import json
with open(OUTPUT_DIR / "problematic_distributions_other.json", "w") as f:
    json.dump(problematic, f, indent=2)
