"""
Unit tests for FactorAnalysis class
"""
from itertools import product

import pytest
import numpy as np

from fa_kit import FactorAnalysis
from fa_kit.factor_analysis import DimensionMismatch, NonSquareMatrix


#
# Testing input validation
#



def test_LabelMatch():

    a_sample = np.random.randn(100,3)

    with pytest.raises(ValueError):
        fan = FactorAnalysis.load_data(a_sample, labels=[0,1])



def test_AssocMatch():

    num_feat = 3
    a_sample = np.random.randn(100, num_feat)
    a_sample -= a_sample.min() - 1

    for m in product([True, False], repeat=2):
        fan = FactorAnalysis.load_data(a_sample, preproc_demean=m[0], preproc_scale=m[1])
        assert fan.data_covar.shape[0] == fan.data_covar.shape[1]
        assert fan.data_covar.shape[0] == num_feat


def test_DimensionMismatch():

    a_sq = np.eye(3)
    b_sq = np.eye(4)

    with pytest.raises(DimensionMismatch):
        fan = FactorAnalysis.load_data_cov(a_sq)
        fan.add_noise_cov(b_sq)


def test_NonSquareCovar():

    a_nonsq = np.ones((4,3))

    with pytest.raises(NonSquareMatrix):
        fan = FactorAnalysis.load_data_cov(a_nonsq)


def test_NonSquareNoise():

    a_sq = np.eye(3)
    b_nonsq = np.ones((4, 3))

    fan = FactorAnalysis.load_data_cov(a_sq)
    with pytest.raises(NonSquareMatrix):
        fan.add_noise_cov(b_nonsq)

#
# Testing extraction
#

TEST_DIM = 100

def test_extraction_covar():

    a_sq = np.eye(TEST_DIM)
    fan = FactorAnalysis.load_data_cov(a_sq)
    fan.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM) / TEST_DIM,
        fan.props_raw
    )

def test_extraction_covar_and_noise():

    a_sq = np.eye(TEST_DIM)
    fan = FactorAnalysis.load_data_cov(a_sq)
    fan.add_noise_cov(a_sq)
    fan.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM) / TEST_DIM,
        fan.props_raw
    )


def test_extraction_data():

    a_sq = np.eye(TEST_DIM)
    a_data = np.concatenate([a_sq]*4, axis=0)

    fan = FactorAnalysis.load_data(a_data)
    fan.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM) / TEST_DIM,
        fan.props_raw
    )

def test_extraction_data_and_noise():

    a_sq = np.eye(TEST_DIM)
    a_data = np.concatenate([a_sq]*4, axis=0)

    fan = FactorAnalysis.load_data(a_data)
    fan.add_noise_cov(a_sq)
    fan.extract_components()

    assert np.array_equal(
        np.ones(TEST_DIM) / TEST_DIM,
        fan.props_raw
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

    fan = FactorAnalysis.load_data_cov(a_cov)
    fan.extract_components()

    return fan


def test_null_retain(random_fa):
    with pytest.raises(Exception):
        random_fa.find_comps_to_retain(method='null')


def test_topn_retain(random_fa, top_n=7):

    random_fa.find_comps_to_retain(method='top_n', num_keep=top_n)

    assert all(random_fa.retain_idx == list(range(top_n)))
