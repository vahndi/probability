from typing import Tuple, overload

from numpy import ndarray
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_generic

from probability.distributions.functions.continuous_function import ContinuousFunction
from probability.distributions.functions.discrete_function import DiscreteFunction


class RVSMixin(object):

    _distribution: rv_generic
    _num_samples: int = 100000

    def rvs(self, num_samples: int) -> ndarray:
        """
        Sample `num_samples` random values from the distribution.
        """
        return self._distribution.rvs(size=num_samples)

    def prob_greater_than(self, other: 'RVSMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) > other.rvs(num_samples)).mean()

    def prob_less_than(self, other: 'RVSMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) < other.rvs(num_samples)).mean()

    def __gt__(self, other: 'RVSMixin'):

        return (self.rvs(self._num_samples) > other.rvs(self._num_samples)).mean()

    def __lt__(self, other: 'RVSMixin'):

        return (self.rvs(self._num_samples) < other.rvs(self._num_samples)).mean()


class MomentMixin(object):

    _distribution: rv_generic

    def moment(self, n: int) -> float:
        """
        n-th order non-central moment of distribution.
        """
        return self._distribution.moment(n=n)


class EntropyMixin(object):

    _distribution: rv_generic

    def entropy(self) -> float:
        """
        Differential entropy of the RV.
        """
        return self._distribution.entropy()


class MedianMixin(object):

    _distribution: rv_generic

    def median(self) -> float:
        """
        Median of the distribution.
        """
        return self._distribution.median()


class MeanMixin(object):

    _distribution: rv_generic

    def mean(self) -> float:
        """
        Median of the distribution.
        """
        return self._distribution.mean()


class StDMixin(object):

    _distribution: rv_generic

    def std(self) -> float:
        """
        Standard deviation of the distribution.
        """
        return self._distribution.std()


class VarMixin(object):

    _distribution: rv_generic

    def var(self) -> float:
        """
        Variance of the distribution.
        """
        return self._distribution.var()


class IntervalMixin(object):

    _distribution: rv_generic

    def interval(self, percent: float) -> Tuple[float, float]:
        """
        Confidence interval with equal areas around the median.
        """
        interval = self._distribution.interval(percent)
        return interval[0], interval[1]


class SupportMixin(object):

    _distribution: rv_generic

    def support(self):
        """
        Return the support of the distribution.
        """
        return self._distribution.support()


class PDFMixin(object):

    _distribution: rv_continuous

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


class PMFMixin(object):

    _distribution: rv_discrete

    def pmf(self) -> DiscreteFunction:
        """
        Probability mass function of the given RV.
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


class CDFMixinC(object):

    _distribution: rv_continuous

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


class CDFMixinD(object):

    _distribution: rv_discrete

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


class SFMixinC(object):

    _distribution: rv_continuous

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
            method_name='logcdf', name='log(CDF)',
            parent=self
        )


class SFMixinD(object):

    _distribution: rv_discrete

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
            method_name='logcdf', name='log(CDF)',
            parent=self
        )


class PPFMixinC(object):

    _distribution: rv_continuous

    def ppf(self) -> ContinuousFunction:
        """
        Percent point function (inverse of cdf) of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='ppf', name='PPF',
            parent=self
        )


class PPFMixinD(object):

    _distribution: rv_discrete

    def ppf(self) -> DiscreteFunction:
        """
        Percent point function (inverse of cdf) of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='ppf', name='PPF',
            parent=self
        )


class ISFMixinC(object):

    _distribution: rv_continuous

    def isf(self) -> ContinuousFunction:
        """
        Inverse survival function (inverse of sf) of the given RV.
        """
        return ContinuousFunction(
            distribution=self._distribution,
            method_name='isf', name='ISF',
            parent=self
        )


class ISFMixinD(object):

    _distribution: rv_discrete

    def isf(self) -> DiscreteFunction:
        """
        Inverse survival function (inverse of sf) of the given RV.
        """
        return DiscreteFunction(
            distribution=self._distribution,
            method_name='isf', name='ISF',
            parent=self
        )
