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
