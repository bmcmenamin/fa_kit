"""
Functions for factor rotation
"""

import numpy as np
import tensorflow as tf

class Rotator(object):
    """
    Base class for rotation objects
    """

    ITER_MAX = 500

    def __init__(self):

        self.comps_orig = None
        self.rot_mat = None
        self.comps_rot = None


    def rotate(self, comps_in):
        """
        Apply a rotation
        """
        raise NotImplementedError

    def flip_to_positive(self):
        """
        Flip to make all abs-max on the positive end
        and unit-norm components
        """

        abs_min = np.abs(np.min(self.comps_rot, axis=0))
        abs_max = np.abs(np.max(self.comps_rot, axis=0))
        to_flip = abs_max < abs_min

        self.comps_rot[:, to_flip] *= -1

        l2_norms = np.sqrt(np.sum(
            self.comps_rot**2,
            axis=0,
            keepdims=True))

        self.comps_rot /= l2_norms



class OrthoRotator(Rotator):
    """
    Class that does orthogonal rotations
    """

    def __init__(self, gamma):

        super(OrthoRotator, self).__init__()
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
        This is one step of the magic rotation computation
        Based on:
        https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
        """

        comps_rot = np.dot(
            self.comps_orig,
            self.rot_mat
            )

        if self.gamma == 0:
            tmp2 = np.dot(
                self.comps_orig.T,
                (comps_rot ** 3)
                )

        else:

            tmp0 = np.diag(
                self.gamma * np.mean(comps_rot ** 2, axis=0)
                )

            tmp1 = np.dot(
                comps_rot,
                tmp0
                )

            tmp2 = np.dot(
                self.comps_orig.T,
                (comps_rot ** 3) - tmp1
                )

        usv = np.linalg.svd(tmp2)

        self.rot_mat = np.dot(usv[0], usv[2])
        score = np.sum(usv[1])

        return score


class VarimaxRotator_python(OrthoRotator):
    """
    class for varimax rotation
    """

    def __init__(self):
        gamma = 1.0
        super(VarimaxRotator_python, self).__init__(gamma)

class QuartimaxRotator_python(OrthoRotator):
    """
    class for varimax rotation
    """

    def __init__(self):
        gamma = 0.0
        super(QuartimaxRotator_python, self).__init__(gamma)




class TfRotator(Rotator):
    """
    Class that does rotations using tensorflow
    """

    def __init__(self):
        super(OrthoRotator, self).__init__()


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
        This is one step of the magic rotation computation
        Based on:
        https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
        """

        comps_rot = np.dot(
            self.comps_orig,
            self.rot_mat
            )

        if self.gamma == 0:
            tmp2 = np.dot(
                self.comps_orig.T,
                (comps_rot ** 3)
                )

        else:

            tmp0 = np.diag(
                self.gamma * np.mean(comps_rot ** 2, axis=0)
                )

            tmp1 = np.dot(
                comps_rot,
                tmp0
                )

            tmp2 = np.dot(
                self.comps_orig.T,
                (comps_rot ** 3) - tmp1
                )

        usv = np.linalg.svd(tmp2)

        self.rot_mat = np.dot(usv[0], usv[2])
        score = np.sum(usv[1])

        return score

class VarimaxRotator_tf(TfRotator):
    """
    class for varimax rotation
    """

    def __init__(self):
        super(VarimaxRotator_tf, self).__init__()

class QuartimaxRotator_tf(TfRotator):
    """
    class for varimax rotation
    """

    def __init__(self):
        super(QuartimaxRotator_tf, self).__init__()