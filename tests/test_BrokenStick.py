import pytest
import numpy as np

from fa_kit.broken_stick import BrokenStick


#
# Testing input validation
#

TEST_DIM = 10


def test_is_sorted():

    list_ascend = list(range(10))
    list_descend = list(range(10))[::-1]

    assert BrokenStick._is_sorted(list_ascend, ascending=True)
    assert BrokenStick._is_sorted(list_descend, ascending=False)
    assert not BrokenStick._is_sorted(list_ascend, ascending=False)
    assert not BrokenStick._is_sorted(list_descend, ascending=True)


def _valid_bs_distro(vals):

    assert len(vals) == TEST_DIM
    assert BrokenStick._is_sorted(vals, ascending=False)


def test_init():

    bs = BrokenStick(TEST_DIM)
    _valid_bs_distro(bs.values)
    assert sum(bs.values) == 1


def test_init_neg():

    with pytest.raises(ValueError):
        bs = BrokenStick(-TEST_DIM)


#
# Testing fit to distro
#


def _fit_to_self(scale, shift):

    orig_vals = BrokenStick._calc_broken_stick(TEST_DIM)
    targ_data = (scale * orig_vals) + shift

    _valid_bs_distro(orig_vals)
    _valid_bs_distro(targ_data)

    bs = BrokenStick.rescale_broken_stick(targ_data)
    _valid_bs_distro(bs.values)

    if shift > 0.0:
        assert np.mean(orig_vals) < np.mean(bs.values)

    if shift < 0.0:
        assert np.mean(orig_vals) > np.mean(bs.values)

    if scale > 1.0:
        assert np.std(orig_vals) < np.std(bs.values)

    if scale < 1.0 and scale > 0.0:
        assert np.std(orig_vals) > np.std(bs.values)


def test_scale_and_fit():

    for scale in [0.5, 1.0, 10.0]:
        _fit_to_self(scale, 0.0)


def test_shift_and_fit():

    for shift in [-0.01, 0.0, 0.01]:
        _fit_to_self(1.0, shift)


#
# Find extremes
#


@pytest.fixture
def shifted_bs():
    orig_vals = BrokenStick._calc_broken_stick(TEST_DIM)
    shift_values = np.concatenate([orig_vals[0::2], -orig_vals[1::2]])
    shift_values = np.array(sorted(shift_values)[::-1])

    bs = BrokenStick.rescale_broken_stick(shift_values)
    _valid_bs_distro(bs.values)

    return bs


def test_find_positive_values(shifted_bs):

    idx_to_find_pos = list(range(2))
    idx_to_find = sorted(idx_to_find_pos)

    targ_values = 1.0 * shifted_bs.values
    targ_values[idx_to_find_pos] += 10.0

    idx_found = shifted_bs.find_where_target_exceeds(targ_values)

    assert idx_found == idx_to_find

def test_find_negative_values(shifted_bs):

    idx_to_find_neg = list(range(TEST_DIM-1, TEST_DIM-3, -1))
    idx_to_find = sorted(idx_to_find_neg)

    targ_values = 1.0 * shifted_bs.values
    targ_values[idx_to_find_neg] -= 10.0

    idx_found = shifted_bs.find_where_target_exceeds(targ_values)

    assert idx_found == idx_to_find

def test_find_positve_and_negative_values(shifted_bs):

    idx_to_find_pos = list(range(2))
    idx_to_find_neg = list(range(TEST_DIM-1, TEST_DIM-3, -1))
    idx_to_find = sorted(idx_to_find_pos + idx_to_find_neg)

    targ_values = 1.0 * shifted_bs.values
    targ_values[idx_to_find_pos] += 10.0
    targ_values[idx_to_find_neg] -= 10.0

    idx_found = shifted_bs.find_where_target_exceeds(targ_values)

    assert idx_found == idx_to_find
