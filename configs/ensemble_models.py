from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier

# CatBoostClassifier does not recognize n_estimators as hyperparameter.
# A workaround is to make n_estimators explicit. See
# https://github.com/scikit-learn/scikit-learn/issues/19844
estimator_dict = {
    "rf": RandomForestClassifier(random_state=10, n_jobs=-1),
    "ada": AdaBoostClassifier(random_state=10),
    "gbm": GradientBoostingClassifier(random_state=10),
    "cat": CatBoostClassifier(n_estimators=1000, random_state=10, verbose=False),
    "lgbm": LGBMClassifier(random_state=10, verbose=-1),
    "xgb": XGBClassifier(random_state=10, n_jobs=-1),
}
