"""
Functions for factor rotation
"""

import numpy as np


class Rotator(object):
    """
    Base class for rotation objects
    """

    ITER_MAX = 5000

    def __init__(self):
        self.comps_orig = None
        self.rot_mat = None
        self.comps_rot = None


    def rotate(self, comps_in):
        """
        Apply iterative orthogonal rotation
        """

        self.comps_orig = comps_in
        self.rot_mat = np.eye(comps_in.shape[1])

        raise NotImplementedError

    def flip_to_positive(self):
        """
        Flip to make all absmax on the positive end
        """

        abs_min = np.abs(np.min(self.comps_rot, axis=0))
        abs_max = np.abs(np.max(self.comps_rot, axis=0))
        to_flip = abs_min < abs_max

        self.comps_rot[:, to_flip] *= -1

        l2_norms = np.sqrt(np.mean(
            self.comps_rot**2,
            axis=0,
            keepdims=True))

        self.comps_rot /= l2_norms



class OrthoRotator(Rotator):
    """
    Class that does orthogonal rotations
    """

    def __init__(self, gamma):

        super(self.__class__).__init__()
        self.gamma = gamma


    def rotate(self, comps_in):
        """
        Apply iterative orthogonal rotation
        """

        self.comps_orig = comps_in
        self.rot_mat = np.eye(comps_in.shape[1])

        score = 0
        for _ in range(self.ITER_MAX):
            old_score, score = score, self._rot_step()
            if score < old_score:
                break

        # apply final rotation
        self.comps_rot = self.comps_orig.dot(self.rot_mat)
        self.flip_to_positive()

        return self.comps_rot


    def _rot_step(self):
        """
        This does all the magic.
        Based on: https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
        """
        term1 = np.dot(
            self.comps_orig,
            self.rot_mat
            )

        term2 = np.diag(np.sum(term1 ** 2, axis=0))
        term2 /= self.comps_orig[0]
        term2 *= self.gamma

        term3 = np.dot(
            self.comps_orig.T,
            term1 ** 3 - np.dot(term1, term2)
            )

        usv = np.linalg.svd(term3)

        self.rot_mat = usv[0].dot(usv[2])
        score = np.sum(usv[1])

        return score


class VarimaxRotator(OrthoRotator):
    """
    class for varimax rotation
    """

    def __init__(self):
        gamma = 1.0
        super(self.__class__).__init__(gamma)

class QuartimaxRotator(OrthoRotator):
    """
    class for varimax rotation
    """

    def __init__(self):
        gamma = 0.0
        super(self.__class__).__init__(gamma)
