import pytest
import numpy as np

from fa_kit import extraction



from fa_kit.broken_stick import BrokenStick


#
# Testing input validation
#

TEST_DIM = 100

def test_is_sorted():

    list_ascend = list(range(10))
    list_descend = list(range(10))[::-1]

    assert extraction._is_sorted(list_ascend, ascending=True)
    assert extraction._is_sorted(list_descend, ascending=False)
    assert not extraction._is_sorted(list_ascend, ascending=False)
    assert not extraction._is_sorted(list_descend, ascending=True)


def test_extraction_covar():
    """
    test extraction of components
    """

    comps, props = extraction.extract_components(np.eye(TEST_DIM))

    assert extraction._is_sorted(props, ascending=False)

    assert np.array_equal(
        np.ones(TEST_DIM)/TEST_DIM,
        props
    )


def test_extraction_covar_and_noise():
    """
    test extraction of components with generalized eig
    """

    comps, props = extraction.extract_components(
        np.eye(TEST_DIM),
        noise_covar=0.1*np.eye(TEST_DIM)
        )

    assert extraction._is_sorted(props, ascending=False)
    
    assert np.array_equal(
        np.ones(TEST_DIM)/TEST_DIM,
        props
    )


def test_paf_step(num_comp=5):

    new_comps, _ = extraction._update_paf(
        num_comp,
        np.ones(TEST_DIM),
        np.eye(TEST_DIM),
        noise_covar=None)
    
    assert np.array_equal(
        new_comps.T.dot(new_comps),
        np.eye(num_comp)
    )

