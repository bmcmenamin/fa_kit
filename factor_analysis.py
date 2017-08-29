"""
The FactorAnalysis object that does most of the work
"""


import numpy as np
import fa_kit as fa

class FactorAnalysis(object):
    """
    Base class for objects extract components
    from data
    """

    def __init__(self, input_data, is_covar=False, noise_covar=None):

        if is_covar:
            if input_data.shape[0] != input_data.shape[1]:
                raise NonSquareMatrix(input_data=input_data)

        if noise_covar is not None:
            if noise_covar.shape[0] != noise_covar.shape[1]:
                raise NonSquareMatrix(noise_covar=noise_covar)

            if noise_covar.shape[1] != input_data.shape[1]:
                raise DimensionMismatch(
                    match_dim=1,
                    noise_covar=noise_covar,
                    input_data=input_data
                    )

        if is_covar:
            self.data_covar = input_data
        else:
            self.data_covar = input_data.T.dot(input_data)

        self.noise_covar = noise_covar

        self.comps_raw = None
        self.comps_paf = None
        self.comps_rot = None

        self.props_raw = None
        self.retain_idx = None


    def extract_components(self):
        """
        decompose data into components
        """

        self.comps_raw, self.props_raw = fa.extraction.extract_components(
            self.data_covar,
            self.noise_covar
            )


    @staticmethod
    def _find_comps_to_retain(props, method='broken_stick', **kwargs):

        if method == 'top_n':
            num_keep = kwargs.get('num_keep', 5)
            retain_idx = fa.retention.retain_top_n(props, num_keep)

        elif method == 'top_pct':
            pct_keep = kwargs.get('pct_keep', .90)
            retain_idx = fa.retention.retain_top_pct(props, pct_keep)

        elif method == 'kaiser':
            data_dim = kwargs.get('data_dim', len(props))
            retain_idx = fa.retention.retain_kaiser(props, data_dim)

        elif method == 'broken_stick':
            retain_idx = fa.retention.retain_broken_stick(props)

        else:
            raise Exception(
                "Unknown method for retention, {}".format(method)
                )

        return retain_idx


    def find_comps_to_retain(self, method='broken_stick', **kwargs):
        """
        Find indices of 'good' components
        default behavior is method='broken_stick' which compares to
        a fitted Broken Stick distribution

        other options:

        top_n: retain the n largest components. requires that you
        pass kwarg num_keep, otherwise n is set to 5

        top_pca: retain however many compenent you need to contain
        the top_pca proportion of all mass

        kaiser: retain the components with absolute values exceeding
        1.0 / data dimensionality. Needs data_dim as input param.

        other
        """

        self.retain_idx = self._find_comps_to_retain(
            self.props_raw,
            method,
            **kwargs
            )



    def reextract_using_paf(self):
        """
        Re-extract the components using "principle axis factoring"
        to downwieght contributions from noise variables and get cleaner
        factors
        """

        self.comps_paf = fa.extraction.extract_using_paf(
            self.comps_raw[:, self.retain_idx],
            self.data_covar,
            noise_covar=self.noise_covar,
            verbose=False
            )


    def rotate_components(self, method='varimax'):
        """
        rotate components
        """

        if method == 'varimax':
            rot_obj = fa.rotation.VarimaxRotator()
        elif method == 'quartimax':
            rot_obj = fa.rotation.QuartimaxRotator()
        else:
            raise Exception(
                "Unknown method for rotation, {}".format(method)
                )

        if self.comps_paf is not None:
            self.comps_rot = rot_obj.rotate(self.comps_paf)
        else:
            self.comps_rot = rot_obj.rotate(self.comps_raw)


    def get_component_scores(self, input_data):
        """
        get component scores on new data
        """

        if self.comps_rot is not None:
            return input_data.dot(self.comps_rot.T)

        if self.comps_paf is not None:
            return input_data.dot(self.comps_paf.T)

        if self.comps_raw is not None:
            return input_data.dot(self.comps_raw.T)

        raise Exception('No components found...')





class NonSquareMatrix(ValueError):
    """
    Exception raised for non-square matrices
    """

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
    """
    Exception raised for mismatched dimensions
    """

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
