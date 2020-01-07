from numpy import ndarray
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_generic
from typing import Tuple

from probability.distributions.functions.continuous_function_1d import ContinuousFunction1d
from probability.distributions.functions.discrete_function_1d import DiscreteFunction1d


class RVS1dMixin(object):

    _distribution: rv_generic
    _num_samples: int = 100000

    def rvs(self, num_samples: int) -> ndarray:
        """
        Sample `num_samples` random values from the distribution.
        """
        return self._distribution.rvs(size=num_samples)

    def prob_greater_than(self, other: 'RVS1dMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) > other.rvs(num_samples)).mean()

    def prob_less_than(self, other: 'RVS1dMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) < other.rvs(num_samples)).mean()

    def __gt__(self, other: 'RVS1dMixin'):

        return (self.rvs(self._num_samples) > other.rvs(self._num_samples)).mean()

    def __lt__(self, other: 'RVS1dMixin'):

        return (self.rvs(self._num_samples) < other.rvs(self._num_samples)).mean()


class Moment1dMixin(object):

    _distribution: rv_generic

    def moment(self, n: int) -> float:
        """
        n-th order non-central moment of distribution.
        """
        return self._distribution.moment(n=n)


class Entropy1dMixin(object):

    _distribution: rv_generic

    def entropy(self) -> float:
        """
        Differential entropy of the RV.
        """
        return self._distribution.entropy()[0]


class Median1dMixin(object):

    _distribution: rv_generic

    def median(self) -> float:
        """
        Median of the distribution.
        """
        return self._distribution.median()


class Mean1dMixin(object):

    _distribution: rv_generic

    def mean(self) -> float:
        """
        Median of the distribution.
        """
        return self._distribution.mean()


class StD1dMixin(object):

    _distribution: rv_generic

    def std(self) -> float:
        """
        Standard deviation of the distribution.
        """
        return self._distribution.std()


class Var1dMixin(object):

    _distribution: rv_generic

    def var(self) -> float:
        """
        Variance of the distribution.
        """
        return self._distribution.var()


class Interval1dMixin(object):

    _distribution: rv_generic

    def interval(self, percent: float) -> Tuple[float, float]:
        """
        Confidence interval with equal areas around the median.
        """
        interval = self._distribution.interval(percent)
        return interval[0], interval[1]


class Support1dMixin(object):

    _distribution: rv_generic

    def support(self):
        """
        Return the support of the distribution.
        """
        return self._distribution.support()


class PDF1dMixin(object):

    _distribution: rv_continuous

    def pdf(self) -> ContinuousFunction1d:
        """
        Probability density function of the given RV.
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='pdf', name='PDF',
            parent=self
        )

    def log_pdf(self) -> ContinuousFunction1d:
        """
        Log of the probability density function of the given RV
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='logpdf', name='log(PDF)',
            parent=self
        )


class PMF1dMixin(object):

    _distribution: rv_discrete

    def pmf(self) -> DiscreteFunction1d:
        """
        Probability mass function of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='pmf', name='PMF',
            parent=self
        )

    def log_pmf(self) -> DiscreteFunction1d:
        """
        Log of the probability mass function of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='logpmf', name='log(PMF)',
            parent=self
        )


class CDF1dMixinC(object):

    _distribution: rv_continuous

    def cdf(self) -> ContinuousFunction1d:
        """
        Cumulative distribution function of the given RV.
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='cdf', name='CDF',
            parent=self
        )

    def log_cdf(self) -> ContinuousFunction1d:
        """
        Log of the cumulative distribution function of the given RV.
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='logcdf', name='log(CDF)',
            parent=self
        )


class CDF1dMixinD(object):

    _distribution: rv_discrete

    def cdf(self) -> DiscreteFunction1d:
        """
        Cumulative distribution function of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='cdf', name='CDF',
            parent=self
        )

    def log_cdf(self) -> DiscreteFunction1d:
        """
        Log of the cumulative distribution function of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='logcdf', name='log(CDF)',
            parent=self
        )


class SF1dMixinC(object):

    _distribution: rv_continuous

    def sf(self) -> ContinuousFunction1d:
        """
        Survival function (1 - cdf) of the given RV.
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='sf', name='SF',
            parent=self
        )

    def log_sf(self) -> ContinuousFunction1d:
        """
        Log of the survival function of the given RV.
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='logcdf', name='log(CDF)',
            parent=self
        )


class SF1dMixinD(object):

    _distribution: rv_discrete

    def sf(self) -> DiscreteFunction1d:
        """
        Survival function (1 - cdf) of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='sf', name='SF',
            parent=self
        )

    def log_sf(self) -> DiscreteFunction1d:
        """
        Log of the survival function of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='logcdf', name='log(CDF)',
            parent=self
        )


class PPF1dMixinC(object):

    _distribution: rv_continuous

    def ppf(self) -> ContinuousFunction1d:
        """
        Percent point function (inverse of cdf) of the given RV.
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='ppf', name='PPF',
            parent=self
        )


class PPF1dMixinD(object):

    _distribution: rv_discrete

    def ppf(self) -> DiscreteFunction1d:
        """
        Percent point function (inverse of cdf) of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='ppf', name='PPF',
            parent=self
        )


class ISF1dMixinC(object):

    _distribution: rv_continuous

    def isf(self) -> ContinuousFunction1d:
        """
        Inverse survival function (inverse of sf) of the given RV.
        """
        return ContinuousFunction1d(
            distribution=self._distribution,
            method_name='isf', name='ISF',
            parent=self
        )


class ISF1dMixinD(object):

    _distribution: rv_discrete

    def isf(self) -> DiscreteFunction1d:
        """
        Inverse survival function (inverse of sf) of the given RV.
        """
        return DiscreteFunction1d(
            distribution=self._distribution,
            method_name='isf', name='ISF',
            parent=self
        )
