"""
Unit tests for BrokenStick class
"""

import pytest
import numpy as np

from fa_kit.broken_stick import BrokenStick


def is_sorted(values, ascending=True):
    """Return True if a sequence is sorted"""

    for i, j in zip(values, values[1:]):

        if ascending and i > j:
            return False

        if not ascending and i < j:
            return False

    return True


#
# Testing input validation
#

TEST_DIM = 10

def _test_valid_bs_distro(vals):
    assert len(vals) == TEST_DIM
    assert is_sorted(vals, ascending=False)

def test_init():
    """ Test the BrokenStick initialized correctly"""
    broken_stick = BrokenStick(TEST_DIM)
    _test_valid_bs_distro(broken_stick.values)
    assert sum(broken_stick.values) == 1

def test_init_neg():
    """ Test the BrokenStick throws an error with bad init inputs"""
    with pytest.raises(ValueError):
        BrokenStick(-TEST_DIM)


#
# Testing fit to distro
#

def _fit_to_self(scale, shift):

    orig_vals = BrokenStick(TEST_DIM).values
    targ_data = (scale * orig_vals) + shift

    _test_valid_bs_distro(orig_vals)
    _test_valid_bs_distro(targ_data)

    broken_stick = BrokenStick(targ_data)
    _test_valid_bs_distro(broken_stick.values)

    if shift > 0.0:
        assert np.mean(orig_vals) < np.mean(broken_stick.values)

    if shift < 0.0:
        assert np.mean(orig_vals) > np.mean(broken_stick.values)

    if scale > 1.0:
        assert np.std(orig_vals) < np.std(broken_stick.values)

    if scale < 1.0 and scale > 0.0:
        assert np.std(orig_vals) > np.std(broken_stick.values)


def test_scale_and_fit():
    """Test that BrokenStick can align to scaled-up/down copies"""
    for scale in [0.5, 1.0, 10.0]:
        _fit_to_self(scale, 0.0)


def test_shift_and_fit():
    """Test that BrokenStick can align to shifted-up/down copies"""
    for shift in [-0.01, 0.0, 0.01]:
        _fit_to_self(1.0, shift)


#
# Find extremes
#

@pytest.fixture
def shifted_bs():
    """Make a BrokenStick thats half positive, half negative"""
    bs_orig = BrokenStick(TEST_DIM)
    orig_vals = bs_orig.values
    shift_values = np.concatenate([orig_vals[0::2], -orig_vals[1::2]])
    shift_values = np.array(sorted(shift_values)[::-1])

    broken_stick = BrokenStick(shift_values)

    return broken_stick


def test_find_positive_values(shifted_bs):
    """Test that we can find extreme positive values"""

    idx_to_find_pos = list(range(2))
    idx_to_find = sorted(idx_to_find_pos)

    targ_values = 1.0 * shifted_bs.values
    targ_values[idx_to_find_pos] += 10.0

    idx_found = shifted_bs.find_where_target_exceeds(targ_values)

    assert idx_found == idx_to_find


def test_find_negative_values(shifted_bs):
    """Test that we can find extreme negative values"""

    idx_to_find_neg = list(range(TEST_DIM-1, TEST_DIM-3, -1))
    idx_to_find = sorted(idx_to_find_neg)

    targ_values = 1.0 * shifted_bs.values
    targ_values[idx_to_find_neg] -= 10.0

    idx_found = shifted_bs.find_where_target_exceeds(targ_values)

    assert idx_found == idx_to_find


def test_find_positve_and_negative_values(shifted_bs):
    """Test that we can find extreme postive and negative values"""

    idx_to_find_pos = list(range(2))
    idx_to_find_neg = list(range(TEST_DIM-1, TEST_DIM-3, -1))
    idx_to_find = sorted(idx_to_find_pos + idx_to_find_neg)

    targ_values = 1.0 * shifted_bs.values
    targ_values[idx_to_find_pos] += 10.0
    targ_values[idx_to_find_neg] -= 10.0

    idx_found = shifted_bs.find_where_target_exceeds(targ_values)

    assert idx_found == idx_to_find
