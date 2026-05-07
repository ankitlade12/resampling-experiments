import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_class(y, prob):
    # threshold dependent metrics
    # Use precision-recall curve to find best threshold
    precisions, recalls, thresholds = precision_recall_curve(y, prob)

    # Calculate F1 scores and get threshold that gives max F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Apply new threshold
    preds = (prob >= best_threshold).astype(int)
    return preds, best_threshold


def evaluate_model_on_test_set(search, X, y):
    # To obtain a dispersion value, we take bootstrapped samples
    # of a portion of the test set.
    y = pd.Series(y, index=X.index)

    roc, ap, precision, recall, thresh = [], [], [], [], []

    for seed in range(1, 6):
        xs = X.sample(frac=0.6, replace=True, random_state=seed)
        ys = y.loc[xs.index]
        prob = search.predict_proba(xs)[:, 1]
        preds, t = predict_class(ys, prob)

        roc.append(roc_auc_score(ys, prob))
        ap.append(average_precision_score(ys, prob))
        precision.append(precision_score(ys, preds))
        recall.append(recall_score(ys, preds))
        thresh.append(t)

    return [
        np.mean(roc),
        np.std(roc),
        np.mean(ap),
        np.std(ap),
        np.mean(precision),
        np.std(precision),
        np.mean(recall),
        np.std(recall),
        np.mean(thresh),
        np.std(thresh),
    ]
