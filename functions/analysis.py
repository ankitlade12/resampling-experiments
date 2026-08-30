import pandas as pd
import warnings


LEGACY_METRICS = [
    "roc",
    "roc_std",
    "ap",
    "ap_std",
    "precision",
    "precision_std",
    "recall",
    "recall_std",
    "f1_score",
    "f1_std",
    "mcc",
    "mcc_std",
    "ba",
    "ba_std",
    "brier",
    "brier_std",
    "gmean",
    "gmean_std",
    "thresh",
    "tresh_std",
]


def holm_adjust(p_values):
    """Adjust a pre-specified family of p-values with Holm's procedure."""
    values = pd.Series(p_values, dtype="float64")
    if values.isna().any() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite values between 0 and 1.")

    order = values.sort_values().index
    m = len(values)
    adjusted_sorted = []
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * values.loc[index])
        adjusted_sorted.append(min(1.0, running_max))

    adjusted = pd.Series(index=values.index, dtype="float64")
    adjusted.loc[order] = adjusted_sorted
    return adjusted


def create_df(scores_dict, dataset, models):
    """
    Create a DataFrame of evaluation scores for a given dataset and set of models.

    Parameters
    ----------
    scores_dict : dict
        Nested dictionary where keys are dataset names and values are
        dictionaries of scores per model, as produced by the evaluation functions.
    dataset : str
        The dataset key to look up in scores_dict.
    models : list of str
        List of model names to include as rows in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with models as rows and evaluation metrics as columns.
        Missing values are rejected so incomplete experiments cannot silently
        appear as zero performance.
    """
    raw = scores_dict[dataset]
    first = next(iter(raw.values()))
    if isinstance(first, dict):
        df = pd.DataFrame.from_dict(raw, orient="index")
        if set(LEGACY_METRICS).issubset(df.columns):
            df = df[LEGACY_METRICS]
    else:
        df = pd.DataFrame(raw, index=LEGACY_METRICS).T

    result = df.loc[models]
    if result.isna().any().any():
        missing = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Missing evaluation values in columns: {missing}")
    return result


def best_performance_summary(
    scores_dict,
    datasets,
    metric,
    metric_std,
    factor = 3,
):
    """
    Build an exploratory summary comparing baselines with their best variant.

    For each dataset and base model, finds the CSL/resampled variant with the highest
    metric value and computes the descriptive difference. Variant selection in
    this function uses evaluation results and therefore must not be interpreted
    as confirmatory model selection or statistical significance:
    - Orange: the single highest metric value across all models for that dataset
      (in either the baseline or CSL/resampled column).
    - Yellow: the selected variant has a positive descriptive difference.

    Parameters
    ----------
    scores_dict : dict
        Nested dict keyed by dataset name, then model name, containing
        evaluation score arrays as produced by the training pipeline.
    datasets : list of str
        Ordered list of dataset names to include in the summary.
    metric : str
        Name of the metric column to compare (e.g. 'roc', 'ap', 'brier').
    metric_std : str
        Name of the corresponding standard deviation column (e.g. 'roc_std').
    factor : int
        Deprecated and retained only for notebook compatibility. Standard
        deviations are not used as significance thresholds.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled DataFrame with one row per (dataset, base model) combination.
    """

    base_models = ["rf", "ada", "gbm", "cat", "lgbm", "xgb"]
    warnings.warn(
        "best_performance_summary selects variants on evaluation results and is "
        "exploratory only. Select variants in nested training CV for final claims.",
        UserWarning,
        stacklevel=2,
    )
    higher_is_better = metric not in {"brier", "log_loss"}

    rows = []
    for data in datasets:
        all_models = list(scores_dict[data].keys())
        df = create_df(scores_dict, data, all_models)
        for model in base_models:
            base_val = df.loc[model, metric]
            base_std = df.loc[model, metric_std]
            variants = [m for m in all_models if m.startswith(model + "_")]
            selector = max if higher_is_better else min
            best_variant = selector(variants, key=lambda v: df.loc[v, metric])
            best_val = df.loc[best_variant, metric]
            best_std = df.loc[best_variant, metric_std]
            diff = (
                best_val - base_val
                if higher_is_better
                else base_val - best_val
            )
            rows.append(
                {
                    "dataset": data,
                    "model": model,
                    metric: base_val,
                    metric_std: base_std,
                    "best_csl_variant": best_variant,
                    f"best_csl_{metric}": best_val,
                    f"best_csl_{metric_std}": best_std,
                    f"{metric}_diff": diff,
                }
            )

    result = pd.DataFrame(rows)

    cols = list(result.columns)
    base_idx = cols.index(metric)
    csl_idx = cols.index(f"best_csl_{metric}")
    diff_idx = cols.index(f"{metric}_diff")

    grouped_values = result.groupby("dataset")[[metric, f"best_csl_{metric}"]]
    dataset_best = (
        grouped_values.max().max(axis=1)
        if higher_is_better
        else grouped_values.min().min(axis=1)
    )

    def style_row(row):
        styles = [""] * len(row)
        diff = row[f"{metric}_diff"]
        if diff > 0:
            styles[diff_idx] = "background-color: yellow"
        best = dataset_best[row["dataset"]]
        if row[metric] == best:
            styles[base_idx] = "background-color: orange"
        elif row[f"best_csl_{metric}"] == best:
            styles[csl_idx] = "background-color: orange"
        return styles

    return result.style.apply(style_row, axis=1)
