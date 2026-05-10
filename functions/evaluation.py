import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def roc_auc_imbalanced(y, prob, n_thresholds=100):
    """
    Compute ROC-AUC using thresholds sampled from the region where probabilities
    cluster, rather than uniformly from [0, 1].

    For imbalanced datasets most predicted probabilities concentrate near zero,
    so uniform threshold spacing misses the informative part of the ROC curve.
    This function finds that region via the 1st–75th percentile of the predicted
    probabilities and samples thresholds densely there, anchored by 0 and 1 to
    ensure the curve starts and ends at (1,1) and (0,0).
    """
    p_low = np.percentile(prob, 1)
    p_high = np.percentile(prob, 75)
    thresholds = np.concatenate([
        [0.0],
        np.linspace(p_low, p_high, n_thresholds),
        [1.0],
    ])

    tprs, fprs = [], []
    for t in thresholds:
        preds = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    fprs, tprs = zip(*sorted(zip(fprs, tprs)))
    return np.trapezoid(tprs, fprs)


def roc_curve_imbalanced(y, prob, n_thresholds=100):
    """
    Compute ROC curve using thresholds sampled from the region where probabilities
    cluster, rather than uniformly from [0, 1].

    For imbalanced datasets most predicted probabilities concentrate near zero,
    so uniform threshold spacing misses the informative part of the ROC curve.
    Thresholds are sampled densely between the 1st and 75th percentile of the
    predicted probabilities, anchored at 0 and 1 to complete the curve.

    Returns
    -------
    fprs : array, sorted ascending
    tprs : array
    thresholds : array
    """
    p_low = np.percentile(prob, 1)
    p_high = np.percentile(prob, 75)
    thresholds = np.concatenate([
        [0.0],
        np.linspace(p_low, p_high, n_thresholds),
        [1.0],
    ])

    tprs, fprs = [], []
    for t in thresholds:
        preds = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    sorted_points = sorted(zip(fprs, tprs, thresholds))
    fprs, tprs, thresholds = zip(*sorted_points)
    return np.array(fprs), np.array(tprs), np.array(thresholds)

# TODO: remove this function
# def precision_recall_curve_imbalanced(y, prob, n_thresholds=100):
#     """
#     Compute precision-recall curve using thresholds sampled from the region
#     where probabilities cluster, rather than uniformly from [0, 1].
#
#     For imbalanced datasets most predicted probabilities concentrate near zero,
#     so uniform threshold spacing misses the informative part of the PR curve.
#     Thresholds are sampled densely between the 1st and 75th percentile of the
#     predicted probabilities, anchored at 0 and 1 to complete the curve.
#     When no positive predictions are made (very high threshold), precision is
#     set to 1.0 following sklearn's convention.
#
#     Returns
#     -------
#     precisions : array, sorted by recall ascending
#     recalls : array
#     thresholds : array
#     """
#     p_low = np.percentile(prob, 1)
#     p_high = np.percentile(prob, 75)
#     thresholds = np.concatenate([
#         [0.0],
#         np.linspace(p_low, p_high, n_thresholds),
#         [1.0],
#     ])
#
#     precisions, recalls = [], []
#     for t in thresholds:
#         preds = (prob >= t).astype(int)
#         tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
#         precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
#         recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
#
#     sorted_points = sorted(zip(recalls, precisions, thresholds))
#     recalls, precisions, thresholds = zip(*sorted_points)
#     return np.array(precisions), np.array(recalls), np.array(thresholds)


def average_precision_imbalanced(y, prob, n_thresholds=100):
    """
    Compute average precision (area under the precision-recall curve) using
    thresholds sampled from the region where probabilities cluster.

    For imbalanced datasets most predicted probabilities concentrate near zero,
    so uniform threshold spacing misses the informative part of the PR curve.
    Thresholds are sampled densely between the 1st and 75th percentile of the
    predicted probabilities, anchored at 0 and 1 to complete the curve.
    When no positive predictions are made (very high threshold), precision is
    set to 1.0 following sklearn's convention.
    """
    p_low = np.percentile(prob, 1)
    p_high = np.percentile(prob, 75)
    thresholds = np.concatenate([
        [0.0],
        np.linspace(p_low, p_high, n_thresholds),
        [1.0],
    ])

    precisions, recalls = [], []
    for t in thresholds:
        preds = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    recalls, precisions = zip(*sorted(zip(recalls, precisions)))
    return np.trapezoid(precisions, recalls)


def predict_class(y, prob):
    # Threshold-dependent metrics:
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
    """
    Evaluate a fitted model on the test set using 5 bootstrap samples (60%,
    with replacement) to estimate dispersion.

    Returns a flat list:
        [mean_roc_auc, std_roc_auc,
         mean_avg_precision, std_avg_precision,
         mean_precision, std_precision,
         mean_recall, std_recall,
         mean_threshold, std_threshold]

    Precision, recall, and threshold are computed at the F1-optimal threshold
    per bootstrap sample.
    """
    # To obtain a dispersion value, we take bootstrapped samples
    # of a portion of the test set.
    X = np.asarray(X)
    y = np.asarray(y)
    n = int(0.6 * len(X))

    roc, ap, precision, recall, thresh = [], [], [], [], []

    for seed in range(1, 6):
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=True)
        xs, ys = X[idx], y[idx]
        prob = search.predict_proba(xs)[:, 1]
        preds, t = predict_class(ys, prob)

        roc.append(roc_auc_imbalanced(ys, prob))
        ap.append(average_precision_imbalanced(ys, prob))
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
