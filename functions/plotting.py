import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from functions.data import load_dataset


def compute_ylim(means, stds, upper, lower, target_fraction=0.28):
    """
    Compute y-axis limits so the RF band (upper - lower) occupies
    approximately target_fraction of the total plot height (between 1/4 and 1/3).
    The axis is always wide enough to show all data points with their error bars.
    """
    band_height = upper - lower
    data_min = (means - stds).min()
    data_max = (means + stds).max()
    data_span = data_max - data_min

    if band_height > 0:
        target_height = band_height / target_fraction
    else:
        target_height = max(data_span * 1.2, 0.01)

    height = max(target_height, data_span * 1.1)
    center = (data_min + data_max) / 2
    return center - height / 2, center + height / 2


def _plot_metrics_on_ax(ax, y, probs, title=None):
    """
    Draw precision, recall and F1 vs threshold on a given ax.
    """
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = (
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    ax.plot(thresholds, precision[:-1], label="Precision", color="blue")
    ax.plot(thresholds, recall[:-1], label="Recall", color="red")
    ax.plot(thresholds, f1_scores, label="F1-Score", color="green")
    ax.axvline(
        best_threshold,
        color="gray",
        linestyle="--",
        lw=1.5,
        label=f"Threshold = {best_threshold:.2f}",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, fontsize=9)


def plot_metrics_vs_threshold_grid(datasets, models, load_model_fn, outputdir=None):
    """
    Plot precision, recall and F1 vs threshold for multiple datasets (rows)
    and models (columns).

    Parameters
    ----------
    datasets : list of str
    models : list of str
    load_model_fn : callable(dataset, model) -> fitted estimator
    """
    n_rows = len(datasets)
    n_cols = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    for i, dataset in enumerate(datasets):
        _, X_test, _, y_test = load_dataset(dataset)

        for j, model in enumerate(models):
            ax = axes[i, j]
            # TODO: I am not sure it's going to work, I need to pass models dir here to load model
            search = load_model_fn(dataset, model)
            probs = search.predict_proba(X_test)[:, 1]

            _plot_metrics_on_ax(ax, y_test, probs, title=f"{dataset}\n{model}")

            if j == 0:
                ax.set_ylabel("Score", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("Threshold", fontsize=9)

    # single legend from the last ax
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle("Precision, Recall and F1 vs Threshold", fontsize=14, y=1.01)
    plt.tight_layout()
    if outputdir is not None:
        plt.savefig(outputdir, dpi=300, bbox_inches="tight")
    plt.show()
