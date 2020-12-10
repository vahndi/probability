from matplotlib.axes import Axes
from scipy.stats import rv_continuous
from typing import Iterable, Optional, Union

from probability.distributions.mixins.plottable_mixin import \
    ContinuousPlottableMixin
from probability.distributions.mixins.rv_mixins import RVS1dMixin, \
    Moment1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin, StD1dMixin, \
    Var1dMixin, Interval1dMixin, Support1dMixin, PDF1dMixin, \
    CDFContinuous1dMixin, SFContinuous1dMixin, PPFContinuous1dMixin, \
    ISFContinuous1dMixin, StatMixin


class RVContinuous1dMixin(
    RVS1dMixin, Moment1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin,
    StD1dMixin, Var1dMixin, Interval1dMixin, Support1dMixin, PDF1dMixin,
    CDFContinuous1dMixin, SFContinuous1dMixin, PPFContinuous1dMixin,
    ISFContinuous1dMixin, ContinuousPlottableMixin, StatMixin, object
):

    _distribution: rv_continuous
    _num_samples: int = 1000000

    def plot(self,
             x: Optional[Iterable] = None,
             kind: str = 'line',
             color: str = 'C0',
             ax: Axes = None, **kwargs) -> Axes:
        """
        Plot the PDF of the distribution.

        :param x: Range of values of x to plot p(x) over.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        return self.pdf().plot(x=x, kind=kind, color=color, ax=ax, **kwargs)

    def __le__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return self.cdf().at(other)
        elif isinstance(other, RVS1dMixin):
            return super(RVContinuous1dMixin, self).__le__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __lt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        return self.__le__(other)

    def __ge__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return 1 - self.__le__(other)
        elif isinstance(other, RVS1dMixin):
            return super(RVContinuous1dMixin, self).__ge__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __gt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        return self.__ge__(other)
