from matplotlib.axes import Axes
from scipy.stats import rv_discrete
from typing import Iterable

from probability.distributions.mixins.rv_mixins import RVS1dMixin, Entropy1dMixin, Median1dMixin, Mean1dMixin, StD1dMixin, \
    Var1dMixin, Interval1dMixin, PMF1dMixin, CDF1dMixinD, SF1dMixinD, PPF1dMixinD, ISF1dMixinD


class RVDiscrete1dMixin(
    RVS1dMixin, Entropy1dMixin, Median1dMixin, Mean1dMixin, StD1dMixin, Var1dMixin, Interval1dMixin,
    PMF1dMixin, CDF1dMixinD, SF1dMixinD, PPF1dMixinD, ISF1dMixinD,
    object
):

    _distribution: rv_discrete
    _num_samples: int = 1000000

    def plot(self, k: Iterable, color: str = 'C0', ax: Axes = None) -> Axes:
        """
        Plot the PMF of the distribution.
        """
        return self.pmf().plot(k=k, color=color, ax=ax)
