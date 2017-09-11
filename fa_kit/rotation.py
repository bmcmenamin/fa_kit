"""Module contains objects used for factor rotation

Currently contains these rotations:
- Varimax (numpy backend)
- Quartimax (numpy backend)
- Pseudo-Varimax (Tensorflow backend)
- Pseudo-Quartimax (Tensorflow backend)

The reason that some are 'pseudo' is that the tensorflow implementation is set
up in a way that optimizes the same criterion as a traditional varimax but does
not guarantee that the rotated components are strictly orthogonal.

The Tensorflow rotation objects make it easy to develop brand-new types of
rotations that optimize other criteria, such as sparseness/L1-regularization.
"""

import numpy as np
import tensorflow as tf

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




class TfRotator(Rotator):
    """Class that does rotations using TensorFlow"""

    def __init__(self):
        super(TfRotator, self).__init__()

        self.graph = None
        self.tf_layers = {}
        self.agg_axis = None


    @staticmethod
    def _new_var(initial_value):
        """Add new variable to model"""

        new_shape = tf.TensorShape(initial_value.shape)

        new_var = tf.Variable(
            initial_value=initial_value,
            expected_shape=new_shape,
            )

        return new_var

    @staticmethod
    def _mse(mat1, mat2):
        """Make variable that is total MSE between two TensorFlow matrices"""

        mse = tf.reduce_mean(
            tf.squared_difference(
                tf.unstack(mat1),
                tf.unstack(mat2),
                name='err'
                ),
            name='mse'
            )

        return mse

    @staticmethod
    def _agg_var(comps, agg_axis):
        """Make variable of varaince in squared component scores on one axis"""

        agg_var = tf.reduce_mean(
            tf.nn.moments(
                comps**2,
                axes=[agg_axis],
                keep_dims=False,
                name='var'
                )[1],
            keep_dims=False,
            name='agg_var'
            )

        return agg_var


    def _build_graph(self):
        """Build graph of for varimax/quartimax rotation"""

        self.graph = tf.Graph()

        with self.graph.as_default():

            self.tf_layers['input'] = tf.placeholder(
                tf.float64,
                shape=self.comps_orig.shape,
                name='input'
                )

            self.tf_layers['comps_orig'] = tf.nn.l2_normalize(
                self.tf_layers['input'],
                0,
                name='comps_orig'
                )

            self.tf_layers['comps_rot'] = tf.nn.l2_normalize(
                self._new_var(self.tf_layers['comps_orig']),
                0,
                name='comps_rot'
                )

            self.tf_layers['comps_recon'] = tf.matmul(
                self.tf_layers['comps_rot'],
                tf.matmul(
                    self.tf_layers['comps_rot'],
                    self.tf_layers['comps_orig'],
                    transpose_a=True,
                    transpose_b=False
                    ),
                transpose_a=False,
                transpose_b=True,
                name='comps_recon'
                )

            self.tf_layers['mse'] = self._mse(
                self.tf_layers['comps_orig'],
                self.tf_layers['comps_recon']
                )

            self.tf_layers['agg_var'] = self._agg_var(
                self.tf_layers['comps_rot'],
                self.agg_axis
                )

            self.tf_layers['optimizer'] = tf.train.AdamOptimizer(
                name='optimizer'
                ).minimize(
                    self.tf_layers['mse'] - self.tf_layers['agg_var']
                    )

    def rotate(self, comps_in):
        """Given a set of components, build a graph and do iterative rotation"""

        self.comps_orig = comps_in
        self._build_graph()

        run_kwargs = {
            'feed_dict': {self.tf_layers['input']: self.comps_orig}
            }
        
        with tf.Session(graph=self.graph) as sess:

            sess.run(
                tf.global_variables_initializer(),
                **run_kwargs
                )

            for _ in range(self.ITER_MAX):
                sess.run(
                    self.tf_layers['optimizer'],
                    **run_kwargs
                    )

            self.comps_rot = sess.run(
                [self.tf_layers['comps_rot']],
                **run_kwargs
                )[0]

        self.flip_to_positive()

        return self.comps_rot


class VarimaxRotatorTf(TfRotator):
    """Tensorflow implementation of pseudo-varimax rotation"""

    def __init__(self):
        super(VarimaxRotatorTf, self).__init__()
        self.agg_axis = 0


class QuartimaxRotatorTf(TfRotator):
    """Tensorflow implementation of pseudo-quartimax rotation"""

    def __init__(self):
        super(QuartimaxRotatorTf, self).__init__()
        self.agg_axis = 1

