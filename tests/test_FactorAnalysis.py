"""
Unit tests for FactorAnalysis class
"""
from itertools import product

import pytest
import numpy as np

from fa_kit import FactorAnalysis
from fa_kit.factor_analysis import DimensionMismatch, NonSquareMatrix


TEST_DIM = 100

A_SAMPLE = np.random.randn(5000, TEST_DIM)

A_SQ = np.eye(TEST_DIM)
B_SQ = np.eye(TEST_DIM + 1)

A_NONSQ = np.ones((TEST_DIM + 1, TEST_DIM))
B_NONSQ = np.ones((TEST_DIM + 1, TEST_DIM))


#
# Testing input validation
#

def test_nonsquare_covar():
    """Test that there's an exception if covar input is not square"""

    with pytest.raises(NonSquareMatrix):
        FactorAnalysis.load_data_cov(A_NONSQ)


def test_nonsquare_noise():
    """Test that there's an exception if noise covar input is not square"""

    fan = FactorAnalysis.load_data_cov(A_SQ)
    with pytest.raises(NonSquareMatrix):
        fan.add_noise_cov(B_NONSQ)


def test_dimension_mismatch():
    """Test that there's an exception if covar and noise matrices are mismatched"""

    with pytest.raises(DimensionMismatch):
        fan = FactorAnalysis.load_data_cov(A_SQ)
        fan.add_noise_cov(B_SQ)


def test_label_match():
    """Throw an error is not the right number of labels"""

    with pytest.raises(ValueError):
        FactorAnalysis.load_data_samples(A_SAMPLE, labels=[0, 1])


def test_assoc_match():
    """Test that output is a square matrix after inputting per-sample data"""

    for preproc_args in product([True, False], repeat=2):

        fan = FactorAnalysis.load_data_samples(
            A_SAMPLE,
            preproc_demean=preproc_args[0],
            preproc_scale=preproc_args[1])

        assert fan.params_data['data_covar'].shape[0] == fan.params_data['data_covar'].shape[1]
        assert fan.params_data['data_covar'].shape[0] == TEST_DIM


#
# Testing getting component scores
#

def test_get_scores():
    """Test getting component scores"""

    num_comps_to_keep = 5
    obs_data = np.random.randn(500, TEST_DIM)

    fan = FactorAnalysis.load_data_cov(A_SQ)
    fan.extract_components()
    fan.find_comps_to_retain(method='top_n', num_keep=num_comps_to_keep)
    fan.reextract_using_paf()

    scores = fan.get_component_scores(obs_data)

    assert scores.shape == (500, num_comps_to_keep)
