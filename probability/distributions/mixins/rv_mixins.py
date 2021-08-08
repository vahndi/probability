from typing import Tuple, Optional, Union, Iterable, List

from numpy import ndarray
from pandas import DataFrame, Series
from scipy.optimize import fmin
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_generic

from probability.distributions.functions.continuous_function_1d import \
    ContinuousFunction1d
from probability.distributions.functions.continuous_function_nd import \
    ContinuousFunctionNd
from probability.distributions.functions.discrete_function_1d import \
    DiscreteFunction1d
from probability.distributions.functions.discrete_function_nd import \
    DiscreteFunctionNd
from probability.distributions.mixins.rv_series import RVSeries, \
    RVContinuousSeries

NUM_SAMPLES_COMPARISON = 100_000


class RVS1dMixin(object):

    _distribution: rv_generic

    def rvs(self, num_samples: int,
            random_state: Optional[int] = None) -> Series:
        """
        Sample `num_samples` random values from the distribution.
        """
        samples = Series(self._distribution.rvs(
            size=num_samples, random_state=random_state
        ))
        samples.index.name = 'sample_index'
        samples.name = str(self)
        return samples

    def prob_less_than(self, other: 'RVS1dMixin',
                       num_samples: int = NUM_SAMPLES_COMPARISON) -> float:

        return (self.rvs(num_samples) < other.rvs(num_samples)).mean()

    def prob_greater_than(self, other: 'RVS1dMixin',
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:

        return (self.rvs(num_samples) > other.rvs(num_samples)).mean()

    def __lt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return (self.rvs(NUM_SAMPLES_COMPARISON) < other).mean()
        elif isinstance(other, RVS1dMixin):
            return (
                self.rvs(NUM_SAMPLES_COMPARISON) <
                other.rvs(NUM_SAMPLES_COMPARISON)
            ).mean()
        else:
            raise TypeError('other must be of type float or Rvs1dMixin')

    def __le__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return (self.rvs(NUM_SAMPLES_COMPARISON) <= other).mean()
        elif isinstance(other, RVS1dMixin):
            return (
                self.rvs(NUM_SAMPLES_COMPARISON) <=
                other.rvs(NUM_SAMPLES_COMPARISON)
            ).mean()
        else:
            raise TypeError('other must be of type float or Rvs1dMixin')

    def __gt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return (self.rvs(NUM_SAMPLES_COMPARISON) > other).mean()
        elif isinstance(other, RVS1dMixin):
            return (
                self.rvs(NUM_SAMPLES_COMPARISON) >
                other.rvs(NUM_SAMPLES_COMPARISON)
            ).mean()
        else:
            raise TypeError('other must be of type float or Rvs1dMixin')

    def __ge__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return (self.rvs(NUM_SAMPLES_COMPARISON) >= other).mean()
        elif isinstance(other, RVS1dMixin):
            return (
                self.rvs(NUM_SAMPLES_COMPARISON) >=
                other.rvs(NUM_SAMPLES_COMPARISON)
            ).mean()
        else:
            raise TypeError('other must be of type float or Rvs1dMixin')


class RVSNdMixin(object):

    _names: List[str]

    def rvs(self, num_samples: int,
            random_state: Optional[int] = None,
            full_name: bool = False) -> DataFrame:
        """
        Sample `num_samples` random values from the distribution.
        """
        if full_name:
            columns = [f'{str(self)}[{name}]' for name in self._names]
        else:
            columns = self._names
        samples = DataFrame(
            data=self._distribution.rvs(size=num_samples,
                                        random_state=random_state),
            columns=columns
        )
        samples.index.name = 'sample_index'
        return samples

    def prob_greater_than(self, other: 'RVSNdMixin',
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> ndarray:

        return (self.rvs(num_samples) > other.rvs(num_samples)).mean(axis=0)

    def prob_less_than(self, other: 'RVSNdMixin',
                       num_samples: int = NUM_SAMPLES_COMPARISON) -> ndarray:

        return (self.rvs(num_samples) < other.rvs(num_samples)).mean(axis=0)

    def __gt__(self, other: 'RVSNdMixin') -> ndarray:

        return (self.rvs(NUM_SAMPLES_COMPARISON) >
                other.rvs(NUM_SAMPLES_COMPARISON)).mean(axis=0)

    def __lt__(self, other: 'RVSNdMixin') -> ndarray:

        return (self.rvs(NUM_SAMPLES_COMPARISON) <
                other.rvs(NUM_SAMPLES_COMPARISON)).mean(axis=0)


class Moment1dMixin(object):

    _distribution: Union[rv_generic, RVSeries]

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
        return self._distribution.entropy()[0]


class Median1dMixin(object):

    _distribution: Union[rv_generic, RVSeries]

    def median(self) -> float:
        """
        Median of the distribution.
        """
        return self._distribution.median()


class Mean1dMixin(object):

    _distribution: Union[rv_generic, RVSeries]

    def mean(self) -> float:
        """
        Mean of the distribution.
        """
        return self._distribution.mean()


class MeanNdMixin(object):

    def mean(self) -> ndarray:
        """
        Mean of the distribution.
        """
        return self._distribution.mean()


class StD1dMixin(object):

    _distribution: Union[rv_generic, RVSeries]

    def std(self) -> float:
        """
        Standard deviation of the distribution.
        """
        return self._distribution.std()


class Var1dMixin(object):

    _distribution: Union[rv_generic, RVSeries]

    def var(self) -> float:
        """
        Variance of the distribution.
        """
        return self._distribution.var()


class VarNdMixin(object):

    def var(self) -> ndarray:
        """
        Variance of the distribution.
        """
        return self._distribution.var()


class Interval1dMixin(object):

    _distribution: Union[rv_generic, RVSeries]

    def interval(self, percent: float) -> Tuple[float, float]:
        """
        Confidence interval with equal areas around the median.
        """
        interval = self._distribution.interval(percent)
        return interval[0], interval[1]


class Support1dMixin(object):

    _distribution: Union[rv_generic, RVSeries]

    def support(self):
        """
        Return the support of the distribution.
        """
        return self._distribution.support()


class PDF1dMixin(object):

    _distribution: Union[rv_continuous, RVContinuousSeries]

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
            method_name='logpdf', name='log PDF',
            parent=self
        )


class PDFNdMixin(object):

    _num_dims: int

    def pdf(self) -> ContinuousFunctionNd:
        """
        Probability mass function of the given RV.
        """
        return ContinuousFunctionNd(
            distribution=self._distribution,
            method_name='pdf', name='PDF',
            num_dims=self._num_dims,
            parent=self
        )

    def log_pdf(self) -> ContinuousFunctionNd:
        """
        Log of the probability mass function of the given RV.
        """
        return ContinuousFunctionNd(
            distribution=self._distribution,
            method_name='logpdf', name='log PDF',
            num_dims=self._num_dims,
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
            method_name='logpmf', name='log PMF',
            parent=self
        )


class PMFNdMixin(object):

    _num_dims: int

    def pmf(self) -> DiscreteFunctionNd:
        """
        Probability mass function of the given RV.
        """
        return DiscreteFunctionNd(
            distribution=self._distribution,
            method_name='pmf', name='PMF',
            num_dims=self._num_dims,
            parent=self
        )

    def log_pmf(self) -> DiscreteFunctionNd:
        """
        Log of the probability mass function of the given RV.
        """
        return DiscreteFunctionNd(
            distribution=self._distribution,
            method_name='logpmf', name='log PMF',
            num_dims=self._num_dims,
            parent=self
        )


class CDFContinuous1dMixin(object):

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
            method_name='logcdf', name='log CDF',
            parent=self
        )


class CDFContinuousNdMixin(object):

    _num_dims: int

    def cdf(self) -> ContinuousFunctionNd:
        """
        Cumulative distribution function of the given RV.
        """
        return ContinuousFunctionNd(
            distribution=self._distribution,
            method_name='cdf', name='CDF',
            num_dims=self._num_dims,
            parent=self
        )

    def log_cdf(self) -> ContinuousFunctionNd:
        """
        Log of the cumulative distribution function of the given RV.
        """
        return ContinuousFunctionNd(
            distribution=self._distribution,
            method_name='logcdf', name='log CDF',
            num_dims=self._num_dims,
            parent=self
        )


class CDFDiscrete1dMixin(object):

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
            method_name='logcdf', name='log CDF',
            parent=self
        )


class SFContinuous1dMixin(object):

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
            method_name='sf', name='log SF',
            parent=self
        )


class SFDiscrete1dMixin(object):

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
            method_name='logsf', name='log SF',
            parent=self
        )


class PPFContinuous1dMixin(object):

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

    def hdi(self, credible_mass: float) -> Tuple[float, float]:
        """
        Return the Highest Density Interval for a given probability mass.

        :param credible_mass: The credible probability mass, from 0 to 1.
        """
        # initial guess for low_tail_prob
        non_credible_mass = 1.0 - credible_mass

        def interval_width(p_low_tail: float):
            return (
                self._distribution.ppf(credible_mass + p_low_tail) -
                self._distribution.ppf(p_low_tail)
            )

        # find low tail probability that minimizes interval width
        p_hdi_low_tail = fmin(interval_width, non_credible_mass,
                              ftol=1e-8, disp=False)[0]

        # return interval as array([low, high])
        low_high = self._distribution.ppf([
            p_hdi_low_tail,
            credible_mass + p_hdi_low_tail
        ])
        return low_high[0], low_high[1]


class PPFDiscrete1dMixin(object):

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


class ISFContinuous1dMixin(object):

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


class ISFDiscrete1dMixin(object):

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


class StatMixin(object):

    def stat(self, stat: Union[str, dict], as_dict: bool):
        """
        Calculate a stat by calling the method on the object.

        :param stat: Name of stat or dict mapping name to iterable of args.
        :param as_dict: Return as a dict with a key as name made from the
                        stat name and arguments.
        """
        if isinstance(stat, str):
            stat_name = stat
            stat_col = stat
            stat_args = ()
        elif isinstance(stat, dict):
            stat_name = list(stat.keys())[0]
            stat_args = list(stat.values())[0]
            if (
                    not isinstance(stat_args, Iterable) or
                    isinstance(stat_args, str)
            ):
                stat_args = (stat_args,)
            stat_col = '__'.join([
                stat_name,
                '_'.join([str(arg) for arg in stat_args])
            ])
        else:
            raise TypeError('Stat must be a str or dict')
        if hasattr(self, stat_name):
            if callable(getattr(self, stat_name)):
                stat_val = getattr(self, stat_name)(*stat_args)
            else:
                stat_val = getattr(self, stat_name)
        else:
            raise ValueError(f'No stat named {stat_name}')
        if as_dict:
            return {stat_col: stat_val}
        else:
            return stat_val
