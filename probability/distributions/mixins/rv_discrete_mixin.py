from typing import Tuple

from numpy import ndarray
from scipy.stats import rv_discrete

from probability.distributions.functions.discrete_function import DiscreteFunction


class RVDiscreteMixin(object):

    _distribution: rv_discrete
    _num_samples: int = 1000000

    def rvs(self, num_samples: int) -> ndarray:
        """
        Sample `num_samples` random values from the distribution.
        """
        return self._distribution.rvs(size=num_samples)

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

    def pmf(self) -> DiscreteFunction:
        """
        Probability mass function at k of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='pmf', name='PMF',
            parent=self
        )

    def log_pmf(self) -> DiscreteFunction:
        """
        Log of the probability mass function of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='logpmf', name='log(PMF)',
            parent=self
        )

    def cdf(self) -> DiscreteFunction:
        """
        Cumulative distribution function of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='cdf', name='CDF',
            parent=self
        )

    def log_cdf(self) -> DiscreteFunction:
        """
        Log of the cumulative distribution function of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='logcdf', name='log(CDF)',
            parent=self
        )

    def sf(self) -> DiscreteFunction:
        """
        Survival function (1 - cdf) of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='sf', name='SF',
            parent=self
        )

    def log_sf(self) -> DiscreteFunction:
        """
        Log of the survival function of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='logsf', name='log(SF)',
            parent=self
        )

    def ppf(self) -> DiscreteFunction:
        """
        Percent point function (inverse of cdf) of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='ppf', name='PPF',
            parent=self
        )

    def isf(self) -> DiscreteFunction:
        """
        Inverse survival function (inverse of sf) of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='isf', name='ISF',
            parent=self
        )

    def prob_greater_than(self, other: 'RVDiscreteMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) > other.rvs(num_samples)).mean()

    def prob_less_than(self, other: 'RVDiscreteMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) < other.rvs(num_samples)).mean()

    def __gt__(self, other: 'RVDiscreteMixin'):

        return (self.rvs(self._num_samples) > other.rvs(self._num_samples)).mean()

    def __lt__(self, other: 'RVDiscreteMixin'):

        return (self.rvs(self._num_samples) < other.rvs(self._num_samples)).mean()
