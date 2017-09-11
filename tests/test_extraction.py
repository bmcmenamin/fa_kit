"""
unit tests for factor extraction module
"""

import pytest
import numpy as np

from fa_kit import extraction
from fa_kit.broken_stick import BrokenStick


def is_sorted(values, ascending=True):
    """Return True if a sequence is sorted"""

    for i, j in zip(values, values[1:]):

        if ascending and i > j:
            return False

        if not ascending and i < j:
            return False

    return True


def test_is_sorted():
    """Make sure that is_sorted works"""

    list_ascend = list(range(10))
    list_descend = list(range(10))[::-1]

    assert is_sorted(list_ascend, ascending=True)
    assert is_sorted(list_descend, ascending=False)
    assert not is_sorted(list_ascend, ascending=False)
    assert not is_sorted(list_descend, ascending=True)




#
# Testing input validation
#

TEST_DIM = 100



def test_extraction_covar():
    """Test extraction of components from covar"""

    comps, props = extraction.extract_components(np.eye(TEST_DIM))

    assert is_sorted(props, ascending=False)

    assert np.allclose(
        np.ones(TEST_DIM) / TEST_DIM,
        props
    )

    assert all(props > 0)


def test_extraction_covar_and_noise():
    """Test extraction of components from covar and noise"""

    comps, props = extraction.extract_components(
        np.eye(TEST_DIM),
        noise_covar=0.2*np.eye(TEST_DIM)
        )

    assert is_sorted(props, ascending=False)
    
    assert np.allclose(
        np.ones(TEST_DIM) / TEST_DIM,
        props
    )


def test_paf_step(num_comp=5):
    """Test extraction of components from covar and noise"""

    new_comps, _ = extraction._paf_step(
        np.eye(TEST_DIM)[:, :num_comp],
        np.eye(TEST_DIM),
        noise_covar=None)
    
    assert np.allclose(
        new_comps.T.dot(new_comps),
        np.eye(num_comp)
    )

