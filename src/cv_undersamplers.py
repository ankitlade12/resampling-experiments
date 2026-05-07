import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.preprocessing import MinMaxScaler


def undersample_data(undersampler, X, y, scale=False, return_index=False):
    """
    Returns 3-fold of undersampled training data with its corresponding
    original left-out fold.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)

    if return_index is True:
        # https://stackoverflow.com/questions/79748461/how-to-pass-pre-computed-folds-to-successivehalving-in-sklearn?
        folds = []

        for fold_train_idx, fold_test_idx in skf.split(X, y):
            if scale is True:
                x = MinMaxScaler().fit_transform(X[fold_train_idx])
            else:
                x = X[fold_train_idx]

            undersampler.fit_resample(x, y[fold_train_idx])

            fold_train_sampled_idx = fold_train_idx[undersampler.sample_indices_]
            folds.append((fold_train_sampled_idx, fold_test_idx))

        return folds

    else:

        xtrain = []
        ytrain = []
        xtest = []
        ytest = []

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            xtrain.append(X.iloc[train_index])
            ytrain.append(y[train_index])
            xtest.append(X.iloc[test_index])
            ytest.append(y[test_index])

        xtrainu = []
        ytrainu = []

        for data, target in zip(xtrain, ytrain):

            if scale is True:
                datau, targetu = undersampler.fit_resample(
                    MinMaxScaler().fit_transform(data),
                    target,
                )
            else:
                datau, targetu = undersampler.fit_resample(data, target)

            xtrainu.append(datau)
            ytrainu.append(targetu)

        # also return undersampled train set for training
        # final model
        if scale is True:
            Xu, yu = undersampler.fit_resample(
                MinMaxScaler().fit_transform(X),
                y,
            )
        else:
            Xu, yu = undersampler.fit_resample(X, y)

        return xtrainu, ytrainu, xtest, ytest, Xu, yu

def train_model_w_undersampling(model, params, xtrainu, ytrainu, xtest, ytest, Xu, yu):
    n_iter = 20  # Number of parameter combinations to try
    param_list = list(ParameterSampler(params, n_iter=n_iter, random_state=42))

    # Store results
    results = []

    # Loop over parameter combinations
    for params_ in param_list:
        fold_scores = []

        # Update with low n_estimators for tuning
        temp_params = params_.copy()
        temp_params['n_estimators'] = 10

        # Cross-validation across the 3 folds
        for i in range(3):
            clf = clone(model)
            clf.set_params(**temp_params)
            clf.fit(xtrainu[i], ytrainu[i])
            y_pred = clf.predict_proba(xtest[i])[:, 1]
            score = roc_auc_score(ytest[i], y_pred)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        results.append((params, avg_score))

    # Get best parameters
    best_params, best_score = max(results, key=lambda x: x[1])

    # Retrain with n_estimators=500
    best_params_final = best_params.copy()
    best_params_final['n_estimators'] = 500

    model = clone(model)
    model.set_params(**best_params_final)
    model.fit(Xu, yu)

    return model

