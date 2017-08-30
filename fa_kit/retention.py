"""
Methods used for calculating factor retention
"""

import numpy as np
from fa_kit.broken_stick import BrokenStick

def retain_top_n(vals, num_keep):
    """
    Retain the top N largest components
    """

    if num_keep < 1:
        raise ValueError(
            "Must select num_keep >= 1 when using 'top_n' retention"
            "criterion. Currently, num_keep = {}".format(num_keep))

    absmag_order = np.argsort(-np.abs(vals))
    retain_idx = absmag_order[:num_keep]

    return retain_idx


def retain_top_pct(vals, pct_keep):
    """
    Retain as many components as you need to capture some proportion
    of the overall absolute eigenvalue distribution
    """

    if pct_keep > 1 or pct_keep <= 0:
        raise ValueError(
            "Must set pct_keep between 0 and 1 be when using "
            "'retain_top_pct' retention criterion. "
            "Currently, pct_keep = {}".format(pct_keep))

    absmag_order = np.argsort(-np.abs(vals))

    cum_pct = 0.0
    retain_idx = []
    for idx in absmag_order:
        if cum_pct < pct_keep:
            retain_idx.append(idx)
            cum_pct += np.abs(vals[idx])
        else:
            break


    return retain_idx


def retain_kaiser(vals, data_dim):
    """
    Use Kaiser's criterion to igure out
    which components to retain based on their
    associated eigenvalues in vals and the data dimension, data_dim
    """

    if data_dim is None or data_dim < len(vals):
        raise ValueError(
            "data_dim is missing or improperly specified "
            "for Kaiser criterion. Current value {}".format(data_dim)
            )

    cutoff_value = 1.0 / data_dim

    retain_idx = [
        key
        for key, val in enumerate(vals)
        if np.abs(val) > cutoff_value
        ]

    return retain_idx


def retain_broken_stick(vals, broken_stick):
    """
    Figure out how many components to keep by aligning
    the dsitribution with a broken stick distribution
    and seeing where your values are larger than expected
    """

    vals = np.array(sorted(vals)[::-1])
    retain_idx = broken_stick.find_where_target_exceeds(vals)

    return retain_idx
