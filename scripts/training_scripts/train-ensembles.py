"""
Train classical machine learning classifiers on various imbalanced datasets.

Models trained: random forests, XGBoost, LightGBM, CatBoost, AdaBoost, and
gradient boosting machines from scikit-learn. Hyperparameter tuning uses
successive halving with the number of trees as the limiting resource.
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
from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from functions.cv import train_model

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)

OUTPUT_DIR = REPO_ROOT / "models" / "classical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

problematic = []
for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    for estimator, params in tqdm(
        zip(estimator_dict, hyperparam_ensemble_dict),
        desc=dataset,
        total=len(estimator_dict),
        leave=False,
    ):
        search = train_model(
            estimator_dict[estimator],
            hyperparam_ensemble_dict[params],
            X_train,
            y_train,
        )
        joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}.pkl")

        n_unique = len(np.unique(search.predict_proba(X_train)[:, 1]))
        if n_unique < 10:
            problematic.append({
                "dataset": dataset,
                "estimator": estimator,
                "n_unique": n_unique,
            })

import json
with open(OUTPUT_DIR / "problematic_distributions.json", "w") as f:
    json.dump(problematic, f, indent=2)
