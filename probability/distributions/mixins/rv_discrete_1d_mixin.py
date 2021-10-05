from typing import Iterable, Optional, Union

from matplotlib.axes import Axes
from scipy.stats import rv_discrete

from probability.custom_types.external_custom_types import FloatArray1d
from probability.distributions.mixins.plottable_mixin import \
    DiscretePlottableMixin
from probability.distributions.mixins.rv_mixins import RVS1dMixin, \
    EntropyMixin, Median1dMixin, Mean1dMixin, StD1dMixin, Var1dMixin, \
    Interval1dMixin, PMF1dMixin, CDFDiscrete1dMixin, SFDiscrete1dMixin, \
    PPFDiscrete1dMixin, ISFDiscrete1dMixin, StatMixin
from probability.utils import is_scalar


class RVDiscrete1dMixin(
    RVS1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin, StD1dMixin,
    Var1dMixin, Interval1dMixin, PMF1dMixin, CDFDiscrete1dMixin,
    SFDiscrete1dMixin, PPFDiscrete1dMixin, ISFDiscrete1dMixin,
    DiscretePlottableMixin, StatMixin, object
):

    _distribution: rv_discrete

    @staticmethod
    def fit(data: FloatArray1d, **kwargs) -> 'RVDiscrete1dMixin':
        """
        Fit a  distribution to the data from a single experiment
        (each experiment represents a series of trials).

        :param data: Iterable of data to fit to.
        """
        raise NotImplementedError

    @staticmethod
    def fits(data: FloatArray1d, **kwargs) -> 'RVDiscrete1dMixin':
        """
        Fit a Binomial distribution to the distribution of results of a series
        of experiments.

        :param data: Iterable of data to fit to.
        """
        raise NotImplementedError

    def plot(self, k: Optional[Iterable] = None,
             kind: str = 'bar',
             color: str = 'C0',
             ax: Optional[Axes] = None,
             **kwargs) -> Axes:
        """
        Plot the PMF of the distribution.

        :param k: Range of values of k to plot p(k) over.
        :param color: Optional color for the series.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        return self.pmf().plot(k=k, kind=kind, color=color, ax=ax, **kwargs)

    def __le__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if is_scalar(other):
            return self.cdf().at(other)
        elif isinstance(other, RVS1dMixin):
            return super(RVDiscrete1dMixin, self).__le__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __lt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if is_scalar(other):
            return self.cdf().at(other - 1)
        elif isinstance(other, RVS1dMixin):
            return super(RVDiscrete1dMixin, self).__lt__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __ge__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if is_scalar(other):
            return 1 - self.cdf().at(other - 1)
        elif isinstance(other, RVS1dMixin):
            return super(RVDiscrete1dMixin, self).__ge__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __gt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if is_scalar(other):
            return 1 - self.cdf().at(other)
        elif isinstance(other, RVS1dMixin):
            return super(RVDiscrete1dMixin, self).__gt__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')
