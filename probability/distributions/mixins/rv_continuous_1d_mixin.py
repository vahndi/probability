from matplotlib.axes import Axes
from scipy.stats import rv_continuous
from typing import Iterable

from probability.distributions.mixins.rv_mixins import RVS1dMixin, Moment1dMixin, Entropy1dMixin, Median1dMixin, Mean1dMixin, StD1dMixin, \
    Var1dMixin, Interval1dMixin, Support1dMixin, PDF1dMixin, CDF1dMixinC, SF1dMixinC, PPF1dMixinC, ISF1dMixinC


class RVContinuous1dMixin(
    RVS1dMixin, Moment1dMixin, Entropy1dMixin, Median1dMixin, Mean1dMixin, StD1dMixin, Var1dMixin, Interval1dMixin, Support1dMixin,
    PDF1dMixin, CDF1dMixinC, SF1dMixinC, PPF1dMixinC, ISF1dMixinC,
    object
):

    _distribution: rv_continuous
    _num_samples: int = 1000000

    def plot(self, x: Iterable, color: str = 'C0', ax: Axes = None) -> Axes:
        """
        Plot the PDF of the distribution.
        """
        return self.pdf().plot(x=x, color=color, ax=ax)
