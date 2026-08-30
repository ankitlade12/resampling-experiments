"""Loaders for additional *hard* imbalanced binary-classification datasets.

The datasets in :mod:`functions.data` are mostly well separated: a single raw
feature, or a standard ensemble, already discriminates the classes very well.
On such data there is little room for resampling to help.

The datasets here were chosen for the opposite reason: standard ensembles
(random forests, XGBoost, CatBoost, LightGBM) achieve only modest performance
because the features do not separate the classes well. They let us test whether
the resampling-based "special" ensembles (RUSBoost, EasyEnsemble,
BalancedRandomForest) add value precisely where ordinary ensembles struggle.

Selection criteria
-------------------
- Naturally binary task (no arbitrary one-vs-rest binarisation of a multiclass
  problem), to avoid the pseudo-replica issue.
- Meaningful class imbalance.
- Hard: no single feature separates the classes (max univariate AUC < 0.90,
  the same screen used in ``notebooks/datasets-with-perfect-separation.ipynb``)
  and the best standard ensemble reaches only a modest ROC-AUC.
- Permissive license: all three are distributed under CC BY 4.0 (commercial use
  permitted with attribution). See ``docs/hard_datasets.md`` for provenance.

Preprocessing is fit on the training partition only. Categorical variables are
ordinal-encoded, constant features are dropped, and ordinary datasets are split
70/30 with stratification. Diabetes encounters are split by patient so that one
patient can never occur in both training and test data. Missing values (absent
from the original datasets but present here) are handled so that the
scikit-learn based estimators can be fitted:
categorical columns get an explicit "Missing" category, and secom's numeric
sensor gaps are flagged out of sample (EndTailImputer at 3x the feature maximum)
rather than mean/median-imputed, so tree models can use "not measured" as a
signal. Every fitted transformation learns from training data only and is then
applied unchanged to test data.
"""

import pickle
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from feature_engine.encoding import OrdinalEncoder
from feature_engine.imputation import EndTailImputer
from feature_engine.selection import DropConstantFeatures
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DATASETS_HARD = ["diabetes130", "default_credit", "secom"]

# UCI Machine Learning Repository ids for the datasets imported via ucimlrepo.
UCI_IDS = {"diabetes130": 296, "default_credit": 350}

DIABETES_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
)

# Patients who died or were discharged to hospice are not eligible for an
# ordinary readmission outcome. These are the exclusions used in the dataset's
# introductory study.
DIABETES_EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

# SECOM is not available through the ucimlrepo API, so it is downloaded directly
# from the UCI static file server (CC BY 4.0).
SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"

CACHE_DIR = Path(__file__).resolve().parent.parent / ".data_cache"


def _cache_path(name):
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def _split(X, y):
    """Create a reproducible stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
        stratify=y,
    )


def _split_by_group(X, y, groups):
    """Split encounters while keeping every patient in exactly one partition."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y[train_idx],
        y[test_idx],
        groups[train_idx],
        groups[test_idx],
    )


def _fetch_uci(name):
    """Fetch (features, targets) from the UCI repo, caching locally."""
    cache = _cache_path(name)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=UCI_IDS[name])
    X = data.data.features.copy()
    y = data.data.targets.copy()
    with open(cache, "wb") as f:
        pickle.dump((X, y), f)
    return X, y


def _fetch_diabetes_with_ids():
    """Fetch the original diabetes table, retaining patient and encounter IDs."""
    cache = _cache_path("diabetes130_with_ids")
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    with urlopen(DIABETES_URL) as resp:
        archive = zipfile.ZipFile(BytesIO(resp.read()))
    with archive.open("diabetic_data.csv") as f:
        data = pd.read_csv(f, na_values="?", low_memory=False)

    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def _fetch_secom():
    """Download (features, labels) for SECOM, caching locally."""
    cache = _cache_path("secom")
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    with urlopen(SECOM_URL) as resp:
        archive = zipfile.ZipFile(BytesIO(resp.read()))
    with archive.open("secom.data") as f:
        X = pd.read_csv(f, sep=r"\s+", header=None, na_values="NaN")
    with archive.open("secom_labels.data") as f:
        labels = pd.read_csv(f, sep=r"\s+", header=None)
    X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    y = labels.iloc[:, 0]  # -1 = pass, 1 = fail; timestamp column is dropped
    with open(cache, "wb") as f:
        pickle.dump((X, y), f)
    return X, y


def _load_diabetes130(return_metadata=False):
    """Hospital readmission (UCI id 296).

    Positive class: patient readmitted within 30 days of discharge ("<30"),
    the clinically meaningful, imbalanced outcome (~11% positive). The other
    outcomes ("NO" and ">30") form the negative class.
    """
    data = _fetch_diabetes_with_ids()
    data = data.loc[
        ~data["discharge_disposition_id"].isin(DIABETES_EXCLUDED_DISPOSITIONS)
    ].reset_index(drop=True)

    target = (data.pop("readmitted") == "<30").astype(int).to_numpy()
    groups = data["patient_nbr"].to_numpy()
    X = data.drop(columns=["encounter_id", "patient_nbr"])

    X_train, X_test, y_train, y_test, groups_train, groups_test = _split_by_group(
        X, target, groups
    )

    categorical = X_train.select_dtypes(include="object").columns.tolist()
    X_train[categorical] = X_train[categorical].fillna("Missing")
    X_test[categorical] = X_test[categorical].fillna("Missing")

    encoder = OrdinalEncoder(
        encoding_method="arbitrary",
        variables=categorical,
        unseen="encode",
    )
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    constant_dropper = DropConstantFeatures()
    X_train = constant_dropper.fit_transform(X_train)
    X_test = constant_dropper.transform(X_test)

    result = (X_train, X_test, y_train, y_test)
    if return_metadata:
        return result + (
            {
                "split_strategy": "patient_grouped",
                "train_groups": groups_train,
                "test_groups": groups_test,
            },
        )
    return result


def _load_default_credit():
    """Credit-card default (UCI id 350).

    Positive class: client defaults on the next payment (~22% positive). All 23
    features are numeric with no missing values, so no encoding is required.
    """
    X, y = _fetch_uci("default_credit")
    target = y.iloc[:, 0].astype(int).to_numpy()
    return _split(X.copy(), target)


def _load_secom():
    """Semiconductor manufacturing yield (UCI id 179).

    Positive class: process failure (~6.6% positive). 590 numeric sensor
    features, most of which are noise; many are constant or have missing values.
    """
    X, y = _fetch_secom()
    target = np.where(y.to_numpy() < 0, 0, 1)  # pass (-1) -> 0, fail (1) -> 1

    X_train, X_test, y_train, y_test = _split(X.copy(), target)
    # Flag sensor NaNs out of sample rather than imputing a central value: for
    # tree models, placing "not measured" far outside the distribution lets the
    # split use the missingness itself. secom has negative values, so we cap at
    # 3x the feature maximum (EndTailImputer) rather than a fixed sentinel.
    imputer = EndTailImputer(imputation_method="max", fold=3)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    constant_dropper = DropConstantFeatures()
    X_train = constant_dropper.fit_transform(X_train)
    X_test = constant_dropper.transform(X_test)
    return X_train, X_test, y_train, y_test


_LOADERS = {
    "diabetes130": _load_diabetes130,
    "default_credit": _load_default_credit,
    "secom": _load_secom,
}


def load_hard_dataset(dataset, return_metadata=False):
    """Load a hard imbalanced dataset by name.

    Parameters
    ----------
    dataset : str
        One of ``DATASETS_HARD``.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Features as DataFrames and binary targets as 1-D integer arrays, split
        70/30 with ``random_state=0`` (matching ``functions.data.load_dataset``).
    """
    if dataset not in _LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {DATASETS_HARD}.")
    if dataset == "diabetes130":
        return _load_diabetes130(return_metadata=return_metadata)

    result = _LOADERS[dataset]()
    if return_metadata:
        return result + ({"split_strategy": "stratified", "test_groups": None},)
    return result
