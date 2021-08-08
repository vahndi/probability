from typing import Tuple

from numpy import inf, ndarray
from pandas import Series
from scipy.stats import moment, gaussian_kde


class RVSeries(object):

    # TODO: EntropyMixin._distribution.entropy

    def __init__(self, data: Series):

        self._data: Series = data.sort_values()

    @property
    def data(self) -> Series:

        return self._data

    def rvs(self, size: int, random_state) -> ndarray:
        """
        Sample `size` random values from the distribution.
        """
        return self._data.sample(
            n=size, replace=True, random_state=random_state
        ).values

    def min(self) -> float:
        """
        Return the smallest value in the data.
        """
        return self._data.min()

    def max(self) -> float:
        """
        Return the largest value in the data.
        """
        return self._data.max()

    def mean(self) -> float:
        """
        Mean of the distribution.
        """
        return self._data.mean()

    def median(self) -> float:
        """
        Median of the distribution.
        """
        return self._data.median()

    def moment(self, n: int) -> float:
        """
        n-th order non-central moment of distribution.
        """
        return moment(a=self._data, moment=n)

    def std(self) -> float:
        """
        Standard deviation of the distribution.
        """
        return self._data.std()

    def var(self) -> float:
        """
        Variance of the distribution.
        """
        return self._data.var()

    def interval(self, percent: float) -> Tuple[float, float]:
        """
        Confidence interval with equal areas around the median.
        """
        lower_pct = 0.5 - percent / 2
        upper_pct = 0.5 + percent / 2
        lower_val, upper_val = self._data.quantile([lower_pct, upper_pct])
        return lower_val, upper_val

    def support(self) -> Tuple[float, float]:
        """
        Return the support of the distribution.
        """
        return -inf, inf


class RVContinuousSeries(RVSeries):

    # TODO: CDFContinuous1dMixin._distribution.cdf
    # TODO: SFContinuous1dMixin._distribution.sf
    # TODO: PPFContinuous1dMixin._distribution.ppf
    # TODO: ISFContinuous1dMixin._distribution.isf

    @property
    def pdf(self):

        kde = gaussian_kde(dataset=self._data)
        return kde.pdf

    @property
    def logpdf(self):

        kde = gaussian_kde(dataset=self._data)
        return kde.logpdf


class RVDiscreteSeries(RVSeries):

    # TODO: PDF1dMixin._distribution.pmf
    # TODO: PDF1dMixin._distribution.logpmf
    pass
