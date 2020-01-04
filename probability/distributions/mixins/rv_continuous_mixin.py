from matplotlib.axes import Axes
from numpy import ndarray
from scipy.stats import rv_continuous
from typing import Tuple, Iterable

from probability.distributions.functions.continuous_function import ContinuousFunction


class RVContinuousMixin(object):

    _distribution: rv_continuous
    _num_samples: int = 1000000

    def rvs(self, num_samples: int) -> ndarray:
        """
        Sample `num_samples` random values from the distribution.
        """
        return self._distribution.rvs(size=num_samples)

    def moment(self, n: int) -> float:
        """
        n-th order non-central moment of distribution.
        """
        return self._distribution.moment(n=n)

    def entropy(self) -> float:
        """
        Differential entropy of the RV.
        """
        return self._distribution.entropy()

    def median(self) -> float:
        """
        Median of the distribution.
        """
        return self._distribution.median()

    def mean(self) -> float:
        """
        Mean of the distribution.
        """
        return self._distribution.mean()

    def std(self) -> float:
        """
        Standard deviation of the distribution.
        """
        return self._distribution.std()

    def var(self) -> float:
        """
        Variance of the distribution.
        """
        return self._distribution.var()

    def interval(self, percent: float) -> Tuple[float, float]:
        """
        Confidence interval with equal areas around the median.
        """
        interval = self._distribution.interval(percent)
        return interval[0], interval[1]

    def support(self):
        """
        Return the support of the distribution.
        """
        return self._distribution.support()

    def pdf(self) -> ContinuousFunction:
        """
        Probability density function of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='pdf', name='PDF',
            parent=self
        )

    def log_pdf(self) -> ContinuousFunction:
        """
        Log of the probability density function of the given RV
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='logpdf', name='log(PDF)',
            parent=self
        )

    def cdf(self) -> ContinuousFunction:
        """
        Cumulative distribution function of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='cdf', name='CDF',
            parent=self
        )

    def log_cdf(self) -> ContinuousFunction:
        """
        Log of the cumulative distribution function of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='logcdf', name='log(CDF)',
            parent=self
        )

    def sf(self) -> ContinuousFunction:
        """
        Survival function (1 - cdf) of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='sf', name='SF',
            parent=self
        )

    def log_sf(self) -> ContinuousFunction:
        """
        Log of the survival function of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='logsf', name='log(SF)',
            parent=self
        )

    def ppf(self) -> ContinuousFunction:
        """
        Percent point function (inverse of cdf) of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='ppf', name='PPF',
            parent=self
        )

    def isf(self) -> ContinuousFunction:
        """
        Inverse survival function (inverse of sf) of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='isf', name='ISF',
            parent=self
        )

    def plot(self, x: Iterable, color: str = 'C0', ax: Axes = None) -> Axes:
        """
        Plot the PDF of the distribution.
        """
        return self.pdf().plot(x=x, color=color, ax=ax)

    def prob_greater_than(self, other: 'RVContinuousMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) > other.rvs(num_samples)).mean()

    def prob_less_than(self, other: 'RVContinuousMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) < other.rvs(num_samples)).mean()

    def __gt__(self, other: 'RVContinuousMixin'):

        return (self.rvs(self._num_samples) > other.rvs(self._num_samples)).mean()

    def __lt__(self, other: 'RVContinuousMixin'):

        return (self.rvs(self._num_samples) < other.rvs(self._num_samples)).mean()
