"""
unit tests for rotation module
"""

import pytest
import numpy as np

from fa_kit import rotation


TEST_DIM = 100

def test_varimax():
    """
    Test varimax rotation
    """

    in_comps = np.eye(TEST_DIM)
    rot = rotation.VarimaxRotator()
    rot_comps = rot.rotate(in_comps)

    assert np.array_equal(
        in_comps.dot(rot_comps.T),
        np.eye(TEST_DIM)
        )

def test_quartimax():
    """
    Test quartimax rotation
    """

    in_comps = np.eye(TEST_DIM)
    rot = rotation.QuartimaxRotator()
    rot_comps = rot.rotate(in_comps)

    assert np.array_equal(
        in_comps.dot(rot_comps.T),
        np.eye(TEST_DIM)
        )
