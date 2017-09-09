"""
The broken-stick object used for calculating the
number of components to retain
"""

import numpy as np

class BrokenStick(object):
    """
    Broken-stick distributed values that can be scaled/shifted
    to align with other distributions
    """

    def __init__(self, in_vals):

        if isinstance(in_vals, (float, int)):
            self.values = self._calc_broken_stick(in_vals)
        else:
            self.values = self._calc_broken_stick(len(in_vals))
            self.rescale_broken_stick(in_vals)

    @staticmethod
    def _calc_broken_stick(dim):
        """
        generate the values for a length "dim" broken
        stick distribution
        """

        if dim < 1:
            raise ValueError('Boken stick dimension must be greater than 1')

        values = np.array([
            1.0 / (i + 1)
            for i in range(dim)
            ])

        values = np.sum(values) - np.insert(np.cumsum(values)[:-1], 0, 0)
        values /= dim

        return values

    @staticmethod
    def _weighted_moments(values, weights):
        """
        returns the mean and standarad deviation of values
        with weights placed on each of the values accoridng to weights
        """

        w_mean = np.average(values, weights=weights)

        sq_err = (values - w_mean)**2
        w_var = np.average(sq_err, weights=weights)

        w_std = np.sqrt(w_var)

        return w_mean, w_std

    @classmethod
    def _fit_to_data(cls, distro_values, target_data, weights=None):
        """
        scale and shift the values in the broken stick to
        best match a set of observed data in target_data

        to account for the purely positive distribution of broken stick values,
        matching is done on log-transformed values and then backprojected to
        the original.

        weights is an array that determines how much weight is placed on
        fitting to each of the values in data.
        """

        targ_log = np.log(target_data + 1)
        dist_log = np.log(distro_values + 1)

        targ_wmean, targ_wsd = cls._weighted_moments(targ_log, weights)
        dist_wmean, dist_wsd = cls._weighted_moments(dist_log, weights)

        scale = targ_wsd / dist_wsd
        shift = targ_wmean - scale*dist_wmean

        dist_log_fit = scale*dist_log + shift

        dist_fit = np.exp(dist_log_fit) - 1

        return dist_fit

    @staticmethod
    def _is_sorted(values, ascending=True):

        for i, j in zip(values, values[1:]):

            if ascending and i > j:
                return False

            if not ascending and i < j:
                return False

        return True

    def rescale_broken_stick(self, target_data, weight_func=lambda x: x**2):
        """
        rescale the broken stick distro's values to align with
        provided target_data. alignment happns by linear shift/scale
        on log-transformed values.

        optional weight_func vector allows you to specify how to downweight
        contributions of individual datapoints to the fit. the weight will
        be calculated by applying this function to the abs-magnitude rank
        of eacn input.

        The default function is np.log, so the largest target_value will
        have weight np.log(1), second largest, np.log(2), etc.

        """


        targ_is_sorted = self._is_sorted(target_data, ascending=False)
        if not targ_is_sorted:
            raise ValueError('Target data is not sorted')

        sort_idx = np.argsort(-np.abs(target_data))
        unsort_idx = np.argsort(sort_idx)

        bs_sorted_fit = self._fit_to_data(
            self.values,
            np.abs(target_data[sort_idx]),
            weight_func(sort_idx + 1)
            )

        bs_unsorted_fit = bs_sorted_fit[unsort_idx] * np.sign(target_data)
        self.values = bs_unsorted_fit


    def find_where_target_exceeds(self, target_data):
        """
        Return the indices where the absolute value of the
        target vector exceed the broken stick distro
        """

        good_idx_pos = []
        good_idx_neg = []

        for idx in range(0, len(target_data)):
            if self.values[idx] > 0 and  target_data[idx] > self.values[idx]:
                good_idx_pos.append(idx)
            else:
                break

        for idx in range(len(target_data) - 1, 0, -1):
            if self.values[idx] < 0 and target_data[idx] < self.values[idx]:
                good_idx_neg.append(idx)
            else:
                break

        all_good_idx = sorted(good_idx_pos + good_idx_neg)

        return all_good_idx
