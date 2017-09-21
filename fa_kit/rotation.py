"""Module contains objects used for factor rotation

Currently contains these rotations:
- Varimax (numpy backend)
- Quartimax (numpy backend)
- Pseudo-Varimax (using Tensorflow backend, in rotation_tf module)
- Pseudo-Quartimax (using Tensorflow backend, in rotation_tf module)

The reason that some are 'pseudo' is that the tensorflow implementation is set
up in a way that optimizes the same criterion as a traditional varimax but does
not guarantee that the rotated components are strictly orthogonal.

The Tensorflow rotation objects make it easy to develop brand-new types of
rotations that optimize other criteria, such as sparseness/L1-regularization.
"""

import numpy as np

class Rotator(object):
    """Base class for rotation objects"""

    ITER_MAX = 100

    def __init__(self):

        self.comps_orig = None
        self.rot_mat = None
        self.comps_rot = None


    def rotate(self, comps_in):
        """Apply a rotation"""
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
    """Class that does orthogonal rotations"""

    def __init__(self, gamma):

        super(OrthoRotator, self).__init__()
        self.gamma = gamma


    def rotate(self, comps_in):

        """
        Apply iterative orthogonal rotation
        based on:
            https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
        """

        self.comps_orig = comps_in
        self.rot_mat = np.eye(self.comps_orig.shape[1])

        var = 0
        for _ in range(self.ITER_MAX):

            lam_rot = np.dot(
                self.comps_orig,
                self.rot_mat
                )

            tmp = np.diag(np.mean(lam_rot ** 2, axis=0) * self.gamma) 

            svd_u, svd_s, svd_v = np.linalg.svd(
                np.dot(
                    self.comps_orig.T,
                    lam_rot ** 3 - np.dot(lam_rot, tmp)
                    )
                )

            self.rot_mat = np.dot(svd_u, svd_v)

            _var = np.sum(svd_s)
            if _var < var:
                break
            var = _var

        # apply final rotation
        self.comps_rot = np.dot(
            self.comps_orig,
            self.rot_mat
            )

        self.flip_to_positive()

        return self.comps_rot


class VarimaxRotatorPython(OrthoRotator):
    """Python implementation of varimax rotation"""

    def __init__(self):
        gamma = 1.0
        super(VarimaxRotatorPython, self).__init__(gamma)


class QuartimaxRotatorPython(OrthoRotator):
    """Python implementation of quartimax rotation"""

    def __init__(self):
        gamma = 0.0
        super(QuartimaxRotatorPython, self).__init__(gamma)
