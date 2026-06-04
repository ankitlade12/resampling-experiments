import numpy as np
from feature_engine.encoding import OrdinalEncoder
from feature_engine.selection import DropConstantFeatures
from imblearn.datasets import fetch_datasets
from keel_ds import load_data
from sklearn.model_selection import train_test_split

DATASETS_IMBLEARN = [
    "abalone_19",
    "arrhythmia",
    "car_eval_4",
    "coil_2000",
    "ecoli",
    "isolet",
    "letter_img",
    "libras_move",
    "mammography",
    "oil",
    "optical_digits",
    "ozone_level",
    "pen_digits",
    "protein_homo",
    "satimage",
    "scene",
    "sick_euthyroid",
    "solar_flare_m0",
    "spectrometer",
    "thyroid_sick",
    "us_crime",
    "webpage",
    "wine_quality",
    "yeast_me2",
]

DATASETS_KEEL = [
    "cleveland-0_vs_4",
    "dermatology-6",
    "glass-0-1-4-6_vs_2",
    "kddcup-buffer_overflow_vs_back",
    "kr-vs-k-one_vs_fifteen",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "page-blocks-1-3_vs_4",
    "pima",
    "poker-8-9_vs_5",
    "shuttle-2_vs_5",
]

DATASETS_LS = DATASETS_IMBLEARN + DATASETS_KEEL

DATASETS_SEPARABLE = [
    "dermatology-6",
    "arrhythmia",
    "kddcup-buffer_overflow_vs_back",
    "kr-vs-k-one_vs_fifteen",
    "shuttle-2_vs_5",
]

DATASETS_FOR_ANALYSIS = [data for data in DATASETS_LS if data not in DATASETS_SEPARABLE]


def load_dataset(dataset):
    if dataset in DATASETS_KEEL:
        # load dataset from keel
        data = load_data(dataset, type_data="imbalanced", raw=True)

        target = data.iloc[:, -1:]
        target = np.where(target == "negative", 0, 1)
        target = target.ravel()

        data = data.iloc[:, :-1]

        if dataset == "kddcup-buffer_overflow_vs_back":
            data = DropConstantFeatures().fit_transform(data)

        # some datasets contain categorical variables
        if dataset in [
            "kddcup-buffer_overflow_vs_back",
            "cleveland-0_vs_4",
            "kr-vs-k-one_vs_fifteen",
        ]:
            data = OrdinalEncoder(encoding_method="arbitrary").fit_transform(data)

        # separate dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            target,
            test_size=0.3,
            random_state=0,
        )

        return X_train, X_test, y_train, y_test

    else:
        # load dataset from imbalanced learn
        data = fetch_datasets()[dataset]
        data.target = np.where(data.target < 0, 0, 1)

        # remove constant features
        if dataset in ["arrhythmia", "oil", "optical_digits", "thyroid_sick"]:
            data.data = DropConstantFeatures().fit_transform(data.data)

        # separate dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.3,
            random_state=0,
        )
        return X_train, X_test, y_train, y_test
