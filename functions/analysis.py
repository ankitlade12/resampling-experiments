import pandas as pd


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
        Missing values are filled with 0.
    """
    df = pd.DataFrame(
      scores_dict[dataset],
      index = ["roc", "roc_std", "ap", "ap_std", "precision", "precision_std",
               "recall", "recall_std", "f1_score", "f1_std", "mcc", "mcc_std",
               "ba", "ba_std", "brier", "brier_std", "gmean", "gmean_std",
               "thresh", "tresh_std"],
    )
    df = df.T
    return df.loc[models].fillna(0)
