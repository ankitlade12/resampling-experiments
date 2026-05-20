"""
Train classifiers with NearMiss-1 undersampling on various imbalanced datasets.

Undersampling is applied once before hyperparameter search to avoid resampling
on every candidate. NearMiss-1 is very aggressive — it keeps only majority class
samples whose average distance to the 3 nearest minority samples is smallest.
Models are saved to disk for later evaluation.
"""
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
import pickle
from tqdm import tqdm

from functions.data import load_dataset, DATASETS_LS
from configs.classical_models import estimator_dict
from configs.hyperparams import hyperparam_classical_dict
from functions.cv_undersamplers import undersample_data, train_model_w_undersampling
from configs.undersamplers import nm1

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)

OUTPUT_DIR = REPO_ROOT / "models" / "nm1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sampling_stats = {}

for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    xtrainu, ytrainu, xtest, ytest, Xu, yu, stats = undersample_data(nm1, X_train, y_train, scale=True)
    sampling_stats[dataset] = stats

    for estimator, params in tqdm(
        zip(estimator_dict, hyperparam_classical_dict),
        desc=dataset,
        total=len(estimator_dict),
        leave=False,
    ):
        search = train_model_w_undersampling(
            estimator_dict[estimator],
            hyperparam_classical_dict[params],
            xtrainu, ytrainu, xtest, ytest, Xu, yu,
        )
        joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}_nm1.pkl")

with open(OUTPUT_DIR / "sampling_stats", "wb") as fp:
    pickle.dump(sampling_stats, fp)
