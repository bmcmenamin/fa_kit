"""Module contains that FactorAnalysis object that coordinates an entire
analysis through a series of in-place data operations.

Typical workflows are illustrated by the notebooks in the `./examples` folder
"""

import numpy as np
import pandas as pd

import fa_kit as fa

from fa_kit.broken_stick import BrokenStick


#
# Custom exceptions
#

class NonSquareMatrix(ValueError):
    """Exception raised for non-square matrices"""

    def __init__(self, **kwargs):

        name = kwargs.keys()[0]
        shape = kwargs[name].shape

        message = (
            'The matrix "{name}" was supposed to be square, but instead '
            'we got something with shape {shape}.'
            ).format(
                name=name,
                shape=shape
            )

        super(NonSquareMatrix, self).__init__(message)


class DimensionMismatch(ValueError):
    """Exception raised for mismatched dimensions"""

    def __init__(self, match_dim=1, **kwargs):

        names = kwargs.keys()
        dims = [kwargs[n].shape[match_dim] for n in names]

        message = (
            'The following matrices supposed to all match on '
            'dimension {match_dim}, but instead we see this: {obs_dims}'
            ).format(
                match_dim=match_dim,
                obs_dims=zip(names, dims)
            )

        super(DimensionMismatch, self).__init__(message)


#
# Utility functions
#

def panda_to_numpy(df_data, labels):
    """ Turn a dataframe into an array with a set of labels in a list"""
    df_data = df_data.select_dtypes(include=[np.number])
    if labels is not None:
        print('Overwriting input labels with column names')
    labels = df_data.columns.tolist()
    np_data = df_data.as_matrix()

    return np_data, labels


def cleanup_labels(np_data, labels):
    """
    Make sure there's the right number of labels. Including making new ones
    from scratch if there aren't any
    """

    if labels is not None:
        if len(labels) != np_data.shape[1]:
            err_text = (
                'Number of labels, {}, does not match '
                'number of data features, {}'
                ).format(len(labels), np_data.shape[1])

            raise ValueError(err_text)
    else:
        labels = list(range(np_data.shape[1]))

    return labels



class FactorAnalysis(object):
    """Base class for FactorAnalysis object"""

    def __init__(self):

        self.params_data = {
            'data_covar': None,
            'noise_covar': None
            }

        self.params_retention = {}
        self.params_rotation = {}

        self.comps = {
            'raw': None,
            'paf': None,
            'rot': None,
            'retain_idx': None
        }

        self.props_raw = None


    def _load_data(self, data, is_cov=False):
        """Function that loads data into factor analysis object"""

        if not isinstance(data, np.ndarray):
            raise TypeError((
                "Input data is not numpy. It's {}".format(type(data))
                ))

        if is_cov:
            if data.shape[0] != data.shape[1]:
                raise NonSquareMatrix(input_data=data)

            self.params_data['input_mean'] = None
            self.params_data['input_scale'] = np.sqrt(
                np.diag(data)
                ).reshape(1, -1)

            if self.params_data['preproc_scale']:
                data /= self.params_data['input_scale']
                data /= self.params_data['input_scale'].T

        else:

            self.params_data['input_mean'] = np.mean(data, axis=0, keepdims=True)

            if self.params_data['preproc_demean']:
                data -= self.params_data['input_mean']

            self.params_data['input_scale'] = np.sqrt(np.mean(
                data**2, axis=0, keepdims=True))

            if self.params_data['preproc_scale']:
                data /= self.params_data['input_scale']

            data = data.T.dot(data) / (data.shape[0] - 1)

        self.params_data['data_covar'] = data


    @classmethod
    def load_data_samples(cls, input_data, labels=None, **kwargs):
        """
        Load an n_samples-by-n_dimensions numpy array or a pandas DataFrame
        into the analysis, and create an n_dimensions-by-n_dimensions
        covariance-esque matrix.

        If you're using an array for input, specify labels for each column
        with the list 'labels'. Labels will be inferred from pandas DataFrames.
        
        Use the boolean flag `preproc_demean` to indicate if you want to demean
        each column before calculating an covariance
        matrix (default: False)

        Use the boolean flag `preproc_scale` to indicate if you want to force
        each column to unit standard deviation before calculating the covariance
        matrix (default: False)
        """

        fa_obj = cls()

        data = input_data.copy()
        if isinstance(data, pd.core.frame.DataFrame):
            data, labels = panda_to_numpy(data, labels)

        labels = cleanup_labels(data, labels)

        new_params = {
            'preproc_demean': kwargs.get('preproc_demean', False),
            'preproc_scale': kwargs.get('preproc_scale', False),
            'labels': labels,
            'labels_dict': {key: val for key, val in enumerate(labels)},
            }

        fa_obj.params_data.update(new_params)

        fa_obj._load_data(data, is_cov=False)

        return fa_obj

    @classmethod
    def load_data_cov(cls, input_data, labels=None, preproc_scale=False):
        """
        Load an n_dimensions-by-n_dimensions numpy array that serves as the
        covariance-esque matrix for analysis.

        Specify labels for each dimension with the list 'labels'.

        Use the boolean flag `preproc_scale` to indicate if you want to force
        each dimension to unit standard deviation by setting the diagonal values
        to 1 (default: False)

        Note: pandas DataFrames are not allowed here.
        """

        fa_obj = cls()

        data = input_data.copy()
        labels = cleanup_labels(data, labels)

        new_params = {
            'preproc_demean': None,
            'preproc_scale': preproc_scale,
            'labels': labels,
            'labels_dict': {key: val for key, val in enumerate(labels)},
            }

        fa_obj.params_data.update(new_params)

        fa_obj._load_data(data, is_cov=True)

        return fa_obj


    def add_noise_cov(self, input_data):
        """
        Load an n_dimensions-by-n_dimensions numpy array that describes the
        distribution of noise.

        Note: pandas not allowed here.
        """

        if not isinstance(input_data, np.ndarray):
            raise TypeError((
                "Input data is not numpy. It's {}".format(type(input_data))
                ))

        if self.params_data['data_covar'] is None:
            raise ValueError("Load data to define self.params_data['data_covar'] first")

        if input_data.shape[0] != input_data.shape[1]:
            raise NonSquareMatrix(input_data=input_data)

        if input_data.shape[1] != self.params_data['data_covar'].shape[1]:
            raise DimensionMismatch(
                match_dim=1,
                noise_covar=input_data,
                input_data=self.params_data['data_covar']
                )

        self.params_data['noise_covar'] = input_data


    def extract_components(self):
        """Extract components"""

        self.comps['raw'], self.props_raw = fa.extraction.extract_components(
            self.params_data['data_covar'],
            self.params_data['noise_covar']
            )


    def find_comps_to_retain(self, method='broken_stick', **kwargs):
        """
        Examine the proportion of variance each component captures and
        determine which to retain. Specificy one of the following methods to use
        with the argument `method` (default `method` = 'broken_stick')

        "top_n": retain the n largest components. requires that you also pass
        the argument `num_keep` (default `num_keep` = 5)

        "top_pct": retain however many compenent you need to capture a certain
        percentage of the overall distibution. Requires that you also pass the
        argument `top_pct` (default `top_pct` = 0.9)

        "kaiser": use Kaiser's criterion to deremine which values are big enough
        to retain based on the dimensionality of the input data

        "broken_stick": fit the observed distribution of proportions to a 
        Broken Stick distribution and see where we have larger-than-expected
        values
        """


        # Store paramteres
        self.params_retention['method'] = method

        if method == 'top_n':
            self.params_retention['num_keep'] = kwargs.get('num_keep', 5)
        elif method == 'top_pct':
            self.params_retention['pct_keep'] = kwargs.get('pct_keep', .90)
        elif method == 'kaiser':
            self.params_retention['data_dim'] = kwargs.get(
                'data_dim', self.params_data['data_covar'].shape[1])
        elif method == 'broken_stick':
            fit_bs_on_log = self.params_data['noise_covar'] is None
            self.params_retention['fit_stick'] = BrokenStick(
                self.props_raw,
                fit_on_log=fit_bs_on_log
                )
        else:
            raise Exception(
                "Unknown method for retention, {}".format(method)
                )


        if method == 'top_n':
            self.comps['retain_idx'] = fa.retention.retain_top_n(
                self.props_raw, self.params_retention['num_keep']
                )

        elif method == 'top_pct':
            self.comps['retain_idx'] = fa.retention.retain_top_pct(
                self.props_raw, self.params_retention['pct_keep']
                )

        elif method == 'kaiser':
            self.comps['retain_idx'] = fa.retention.retain_kaiser(
                self.props_raw, self.params_retention['data_dim']
                )

        elif method == 'broken_stick':
            self.comps['retain_idx'] = fa.retention.retain_broken_stick(
                self.props_raw, self.params_retention['fit_stick']
                )

        return self.comps['retain_idx']


    def reextract_using_paf(self):
        """
        Re-extract the components using "Principle Axis Factoring"
        to downwieght contributions from noisy variables.

        Must be run after `extract_components` and `find_comps_to_retain`
        """

        self.comps['paf'] = fa.extraction.extract_using_paf(
            self.comps['raw'][:, self.comps['retain_idx']],
            self.params_data['data_covar'],
            noise_covar=self.params_data['noise_covar'],
            verbose=False
            )


    def rotate_components(self, method='varimax'):
        """
        Rotate extracted components by on of these types as the argument
        'method' (default `method` = 'varimax')

          `varimax`, `quartimax` (numpy-based orthogonal rotations)
          `varimax_tf`, `quartimax_tf` (tensorflow-based rotations)

        Must be run after either `extract_components` or `reextract_using_paf`
        """

        self.params_rotation['method'] = method

        if method == 'varimax':
            rot_obj = fa.rotation.VarimaxRotatorPython()
        elif method == 'varimax_tf':
            rot_obj = fa.rotation_tf.VarimaxRotatorTf()
        elif method == 'quartimax':
            rot_obj = fa.rotation.QuartimaxRotatorPython()
        elif method == 'quartimax_tf':
            rot_obj = fa.rotation_tf.QuartimaxRotatorTf()
        else:
            raise Exception(
                "Unknown method for rotation, {}".format(method)
                )

        if self.comps['paf'] is not None:
            self.comps['rot'] = rot_obj.rotate(self.comps['paf'])
        elif self.comps['raw'] is not None:
            self.comps['rot'] = rot_obj.rotate(
                self.comps['raw'][:, self.comps['retain_idx']]
                )
        else:
            print('You must extract components before rotations')


    def get_component_scores(self, input_data):
        """Project samples onto components to get component scores per sample"""

        # Apply preprocessing
        if self.params_data['preproc_demean']:
            input_data -= self.params_data['input_mean']

        if self.params_data['preproc_scale']:
            input_data /= self.params_data['input_scale']

        # Project data onto components
        if self.comps['rot'] is not None:
            return input_data.dot(self.comps['rot'])

        if self.comps['paf'] is not None:
            return input_data.dot(self.comps['paf'])

        if self.comps['raw'] is not None:
            return input_data.dot(self.comps['raw'])

        raise Exception('No components found in model. Run extraction.')
