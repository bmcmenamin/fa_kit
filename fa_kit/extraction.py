"""
Functions for component extraction
"""

import numpy as np
from scipy import linalg as sp_linalg

import fa_kit.retention as retention


PAF_OPTS = {
    'max_iter': 100,
    'tol': 1.0e-4
}

def _is_sorted(values, ascending=True):

    for i, j in zip(values, values[1:]):

        if ascending and i > j:
            return False

        if not ascending and i < j:
            return False

    return True

def _ensure_ordering(comps, props):
    """
    ensures that the extract components
    are properly ordered
    """

    new_order = np.argsort(props)[::-1]

    comps = comps[:, new_order]
    props = props[new_order]

    return comps, props


def extract_components(data_covar, noise_covar=None):
    """
    Extract components from a covariance matrix, data_covar
    If an additional covar matrix, noise_covar, is added we will
    use generalized eigenvalue solution
    """

    if noise_covar is not None:
        props, comps = sp_linalg.eigh(
            a=data_covar,
            b=noise_covar
            )
    else:
        props, comps = np.linalg.eigh(
            data_covar
            )

    comps = np.real(comps)
    props = np.real(props)
    props /= np.sum(np.abs(props))

    comps, props = _ensure_ordering(comps, props)

    return comps, props


def _update_paf(num_comp, communality, data_covar, noise_covar=None):

    modified_covar = np.copy(data_covar)

    np.fill_diagonal(
        modified_covar,
        communality * np.diag(data_covar)
        )

    new_comps, new_props = extract_components(
        modified_covar,
        noise_covar
        )

    retain_idx = retention.retain_top_n(new_props, num_comp)

    return new_comps[:, retain_idx], new_props


def extract_using_paf(comps, data_covar, noise_covar=None, verbose=False):
    """
    use principle axis factoring
    """

    new_comps = np.copy(comps)
    new_props = np.zeros(new_comps.shape[0])

    for step in range(PAF_OPTS['max_iter']):

        old_comps, old_props = new_comps, new_props

        new_comps, new_props = _update_paf(
            old_comps.shape[1],
            np.sum(old_comps**2, axis=1),
            data_covar,
            noise_covar=noise_covar)

        err = np.mean((new_props - old_props)**2)

        if verbose:
            print("Iteration {} error: {}".format(step, err))

        if err < PAF_OPTS['tol']:
            break

    return new_comps
