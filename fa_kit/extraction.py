"""This module contains functions used for extracting components
from a covariance matrix.
"""

import numpy as np
import scipy as sp

import fa_kit.retention as retention


# Options used in Factor extraction
EIG_OPTS = {
    'noise_reg': 1.0e-4
}

# Options used in Prinicple Axis Factoring
PAF_OPTS = {
    'max_iter': 100,
    'tol': 1.0e-4
}


def reorder_components(comps, props):
    """Ensure that the components are ordered in terms of decreasing proportion"""

    new_order = np.argsort(props)[::-1]

    comps = comps[:, new_order]
    props = props[new_order]

    return comps, props


def extract_components(data_covar, noise_covar=None):
    """
    Given a covariance matrix, `data_covar`, extract it's components (`comps`)
    and the proportion of covaraince that each explains (`props`).

    You can also specify a second matrix `noise_covar` that contains
    information about how noise is distributed. If you do that, `props` will
    be replaced by the log-SNR value which can be negative.
    """

    if noise_covar is None:

        # Extract component using eigendecomposition
        props, comps = np.linalg.eigh(
            data_covar
            )

    else:
        # Add an identity matrix to the noise matrix to ensure full rank
        reg_noise = EIG_OPTS['noise_reg']*np.eye(noise_covar.shape[0])
        reg_noise += noise_covar

        inv_noise = np.linalg.inv(reg_noise)

        # Pre-multiply by the inverse nose matrix, which is equivalent to
        # doing a generalized eigenvalue solution
        props, comps = np.linalg.eigh(
            inv_noise.dot(data_covar)
            )

        # Turn eigenvalues into log-SNR
        base_snr = np.trace(data_covar) / np.trace(reg_noise)
        props = np.log(props) + np.log(base_snr)

    comps = np.real(comps)
    props = np.real(props)
    props /= np.sum(np.abs(props))

    comps, props = reorder_components(comps, props)

    return comps, props


def _paf_step(comps, data_covar, noise_covar=None):
    """Re-extract components while downweighting by communality"""

    communality = np.sum(comps**2, axis=1)

    # Use communality to scale down the variance along the diagonal
    modified_covar = np.copy(data_covar)
    np.fill_diagonal(
        modified_covar,
        communality * np.diag(data_covar)
        )

    new_comps, new_props = extract_components(
        modified_covar,
        noise_covar
        )

    retain_idx = retention.retain_top_n(new_props, comps.shape[1])

    return new_comps[:, retain_idx], new_props


def extract_using_paf(comps, data_covar, noise_covar=None, verbose=False):
    """
    Use principle axis factoring to re-extract factors from a covariance
    matrix while reducing the weights placed on features with low communality
    """

    new_comps = np.copy(comps)
    new_props = np.zeros(new_comps.shape[0])

    for step in range(PAF_OPTS['max_iter']):

        old_comps, old_props = new_comps, new_props

        new_comps, new_props = _paf_step(
            old_comps,
            data_covar,
            noise_covar=noise_covar)

        err = np.mean((new_props - old_props)**2)

        if verbose:
            print("Iteration {} error: {}".format(step, err))

        if err < PAF_OPTS['tol']:
            break

    return new_comps
