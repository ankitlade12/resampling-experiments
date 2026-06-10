"""
Train ensemble (tree-based) classifiers on the *hard* imbalanced datasets.

Same models and tuning as scripts/training_scripts/train-ensembles.py (random
forests, XGBoost, LightGBM, CatBoost, AdaBoost and gradient boosting; successive
halving with the number of trees as the resource), but on the datasets in
functions.hard_data, where standard ensembles perform poorly.
"""

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json

import joblib
import numpy as np
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from functions.cv import train_model
from functions.hard_data import DATASETS_HARD, load_hard_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

OUTPUT_DIR = REPO_ROOT / "models" / "ensembles-hard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

problematic = []
for dataset in tqdm(DATASETS_HARD, desc="Datasets"):
    X_train, X_test, y_train, y_test = load_hard_dataset(dataset)

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
            scoring="roc_auc",
        )
        joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}.pkl")

        n_unique = len(np.unique(search.predict_proba(X_train)[:, 1]))
        if n_unique < 10:
            problematic.append(
                {
                    "dataset": dataset,
                    "estimator": estimator,
                    "n_unique": n_unique,
                }
            )

with open(OUTPUT_DIR / "problematic_distributions.json", "w") as f:
    json.dump(problematic, f, indent=2)
