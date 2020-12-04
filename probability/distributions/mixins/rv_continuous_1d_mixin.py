from matplotlib.axes import Axes
from scipy.stats import rv_continuous
from typing import Iterable, Optional

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
