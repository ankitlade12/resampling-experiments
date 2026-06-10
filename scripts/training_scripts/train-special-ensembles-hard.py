"""
Train special ensemble classifiers on the *hard* imbalanced datasets.

Models trained: Balanced Random Forest, Easy Ensemble and RUSBoost -- ensemble
methods that incorporate resampling into their design. Same tuning as
scripts/training_scripts/train-special-ensembles.py (successive halving with the
number of trees as the resource), but on the datasets in functions.hard_data.

Unlike the original script, no per-dataset sampling_strategy override is needed:
the minority classes here are large enough for the default "auto" strategy.
"""

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from tqdm import tqdm

from configs.hyperparams import hyperparam_special_dict
from configs.special_ensembles import estimator_dict
from functions.cv import train_model
from functions.hard_data import DATASETS_HARD, load_hard_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="The total space of parameters")
warnings.simplefilter(action="ignore", category=FutureWarning)

OUTPUT_DIR = REPO_ROOT / "models" / "special-ensembles-hard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for dataset in tqdm(DATASETS_HARD, desc="Datasets"):
    X_train, X_test, y_train, y_test = load_hard_dataset(dataset)

    for estimator, params in tqdm(
        zip(estimator_dict, hyperparam_special_dict),
        desc=dataset,
        total=len(estimator_dict),
        leave=False,
    ):
        search = train_model(
            estimator_dict[estimator],
            hyperparam_special_dict[params],
            X_train,
            y_train,
            scoring="roc_auc",
        )
        joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}.pkl")
