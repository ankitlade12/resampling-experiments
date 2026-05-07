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
    x1 = X.sample(frac=0.6, replace=True, random_state=1)
    x2 = X.sample(frac=0.6, replace=True, random_state=2)
    x3 = X.sample(frac=0.6, replace=True, random_state=3)

    # Adjust the target
    y = pd.Series(y, index=X.index)

    y1 = y.loc[x1.index]
    y2 = y.loc[x2.index]
    y3 = y.loc[x3.index]

    # Make predictions on test sets
    prob1 = search.predict_proba(x1)[:, 1]
    prob2 = search.predict_proba(x2)[:, 1]
    prob3 = search.predict_proba(x3)[:, 1]

    # threshold independent metrics
    roc = [roc_auc_score(y1, prob1), roc_auc_score(y2, prob2), roc_auc_score(y3, prob3)]
    ap = [
        average_precision_score(y1, prob1),
        average_precision_score(y2, prob2),
        average_precision_score(y3, prob3),
    ]

    # calculate class based on optimal threshold
    preds1, thresh1 = predict_class(y1, prob1)
    preds2, thresh2 = predict_class(y2, prob2)
    preds3, thresh3 = predict_class(y3, prob3)

    # Calculate precision and recall
    precision = [
        precision_score(y1, preds1),
        precision_score(y2, preds2),
        precision_score(y3, preds3),
    ]
    recall = [
        recall_score(y1, preds1),
        recall_score(y2, preds2),
        recall_score(y3, preds3),
    ]

    thresh = [thresh1, thresh2, thresh3]

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
