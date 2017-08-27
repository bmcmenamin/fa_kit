import pytest
import numpy as np

from fa_kit import FactorAnalysis
from fa_kit.factor_analysis import DimensionMismatch, NonSquareMatrix


#
# Testing input validation
#

def test_DimensionMismatch():

    a_sq = np.eye(3)
    b_sq = np.eye(4)

    with pytest.raises(DimensionMismatch):
        FactorAnalysis(a_sq, noise_covar=b_sq)


def test_NonSquareCovar():

    a_nonsq = np.ones((4,3))

    with pytest.raises(NonSquareMatrix):
        FactorAnalysis(a_nonsq, is_covar=True)


def test_NonSquareNoise():

    a_sq = np.eye(3)
    b_nonsq = np.ones((4, 3))

    with pytest.raises(NonSquareMatrix):
        FactorAnalysis(a_sq, noise_covar=b_nonsq)

#
# Testing extraction
#

TEST_DIM = 100

def test_extraction_covar():

    a_sq = np.eye(TEST_DIM)
    test_analysis = FactorAnalysis(
        a_sq,
        is_covar=True)
    test_analysis.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM)/TEST_DIM,
        test_analysis.props_raw
    )

def test_extraction_covar_and_noise():

    a_sq = np.eye(TEST_DIM)
    test_analysis = FactorAnalysis(
        a_sq,
        noise_covar=a_sq,
        is_covar=True)
    test_analysis.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM)/TEST_DIM,
        test_analysis.props_raw
    )


def test_extraction_data():

    a_sq = np.eye(TEST_DIM)
    a_data = np.concatenate([a_sq]*4, axis=0)

    test_analysis = FactorAnalysis(
        a_data,
        is_covar=False)
    test_analysis.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM)/TEST_DIM,
        test_analysis.props_raw
    )

def test_extraction_data_and_noise():

    a_sq = np.eye(TEST_DIM)
    a_data = np.concatenate([a_sq]*4, axis=0)

    test_analysis = FactorAnalysis(
        a_data,
        noise_covar=a_sq,
        is_covar=False)
    test_analysis.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM)/TEST_DIM,
        test_analysis.props_raw
        )

#
# Testing number to retain calls
#

@pytest.fixture
def random_fa():
    a_data = np.random.randn(10000, TEST_DIM)

    # adding correltions between vars by superimposing
    # the same pattern of random noise over different
    # variable ranges

    bin_width = 15
    step_size = 10
    for idx_start in range(0, TEST_DIM, step_size):
        new_noise = 0.2 * np.random.randn(10000, 1)
        a_data[:, idx_start:(idx_start+bin_width)] += new_noise

    a_cov = a_data.T.dot(a_data)

    test_analysis = FactorAnalysis(
        a_cov,
        is_covar=True)

    test_analysis.extract_components()

    return test_analysis


def test_null_retain(random_fa):
    with pytest.raises(Exception):
        random_fa.find_comps_to_retain(method='null')


def test_topn_retain(random_fa, top_n=7):

    random_fa.find_comps_to_retain(method='top_n', num_keep=top_n)

    assert all(random_fa.retain_idx == list(range(top_n)))
