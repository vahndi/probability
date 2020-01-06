from matplotlib.axes import Axes
from scipy.stats import rv_discrete
from typing import Iterable

from probability.distributions.mixins.rv_mixins import RVSMixin, EntropyMixin, MedianMixin, MeanMixin, StDMixin, \
    VarMixin, IntervalMixin, PMFMixin, CDFMixinD, SFMixinD, PPFMixinD, ISFMixinD


class RVDiscreteMixin(
    RVSMixin, EntropyMixin, MedianMixin, MeanMixin, StDMixin, VarMixin, IntervalMixin,
    PMFMixin, CDFMixinD, SFMixinD, PPFMixinD, ISFMixinD,
    object
):

    _distribution: rv_discrete
    _num_samples: int = 1000000

    def plot(self, k: Iterable, color: str = 'C0', ax: Axes = None) -> Axes:
        """
        Plot the PMF of the distribution.
        """
        return self.pmf().plot(k=k, color=color, ax=ax)
