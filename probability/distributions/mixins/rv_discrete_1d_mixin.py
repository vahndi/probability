from matplotlib.axes import Axes
from scipy.stats import rv_discrete
from typing import Iterable

from probability.distributions.mixins.plottable_mixin import PlottableMixin
from probability.distributions.mixins.rv_mixins import RVS1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin, \
    StD1dMixin, Var1dMixin, Interval1dMixin, PMF1dMixin, CDFDiscrete1dMixin, SFDiscrete1dMixin, PPFDiscrete1dMixin, \
    ISFDiscrete1dMixin


class RVDiscrete1dMixin(
    RVS1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin, StD1dMixin, Var1dMixin, Interval1dMixin,
    PMF1dMixin, CDFDiscrete1dMixin, SFDiscrete1dMixin, PPFDiscrete1dMixin, ISFDiscrete1dMixin,
    PlottableMixin, object
):

    _distribution: rv_discrete
    _num_samples: int = 1000000

    def plot(self, k: Iterable, kind: str = 'bar', color: str = 'C0', ax: Axes = None,
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
