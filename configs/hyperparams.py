from scipy import stats

hyperparam_classical_dict = {
    "rf_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "min_samples_split": stats.randint(2, 20),
        "min_samples_leaf": stats.randint(2, 20),
        "max_features": ("log2", 0.25, "sqrt", 1.0),
    },
    "ada_params": {
        "learning_rate": stats.uniform(0, 1),
    },
    "gbm_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "learning_rate": stats.uniform(0, 1),
        "min_samples_split": stats.randint(2, 20),
        "max_features": ("log2", 0.25, "sqrt", 1.0),
        "subsample": stats.uniform(0, 1),
    },
    "cat_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "learning_rate": stats.uniform(0, 1),
        "leaf_estimation_iterations": stats.randint(1, 10),
        "l2_leaf_reg": stats.randint(1, 10),
    },
    "lgbm_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "learning_rate": stats.uniform(0, 1),
        "num_leaves": stats.randint(1, 1024),
        "colsample_bytree": stats.uniform(0, 1),
    },
    "xgb_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "learning_rate": stats.uniform(0, 1),
        "gamma": stats.uniform(0, 2),
        "colsample_bytree": stats.uniform(0, 1),
        "colsample_bynode": stats.uniform(0, 1),
        "colsample_bylevel": stats.uniform(0, 1),
    },
}

hyperparam_special_dict = {
    "rus_params": {
        "learning_rate": stats.uniform(0, 10),
    },
    "easy_params": {
        "replacement": (True, False),
    },
    "brf_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "min_samples_split": stats.randint(2, 20),
        "min_samples_leaf": stats.randint(2, 20),
        "max_features": ("log2", 0.25, "sqrt", 1.0),
    },
}


# TODO: remove function, seems to be unused
def append_model_prefix(param_dict):
    """
    Append 'model__' prefix to all keys in each subdictionary
    """
    result = {}

    for model_name, params in param_dict.items():
        prefixed_params = {}
        for key, value in params.items():
            prefixed_params[f"model__{key}"] = value
        result[model_name] = prefixed_params

    return result
