from matplotlib.axes import Axes
from numpy import ndarray
from scipy.stats import rv_continuous
from typing import Tuple, Iterable

from probability.distributions.functions.continuous_function import ContinuousFunction
from probability.distributions.mixins.rv import RVSMixin, MomentMixin, EntropyMixin, MedianMixin, MeanMixin, StDMixin, \
    VarMixin, IntervalMixin, SupportMixin, PDFMixin, CDFMixinC, SFMixinC, PPFMixinC, ISFMixinC


class RVContinuousMixin(
    RVSMixin, MomentMixin, EntropyMixin, MedianMixin, MeanMixin, StDMixin, VarMixin,
    IntervalMixin, SupportMixin, PDFMixin, CDFMixinC, SFMixinC, PPFMixinC, ISFMixinC,
    object
):

    _distribution: rv_continuous
    _num_samples: int = 1000000

    def plot(self, x: Iterable, color: str = 'C0', ax: Axes = None) -> Axes:
        """
        Plot the PDF of the distribution.
        """
        return self.pdf().plot(x=x, color=color, ax=ax)

    def prob_greater_than(self, other: 'RVContinuousMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) > other.rvs(num_samples)).mean()

    def prob_less_than(self, other: 'RVContinuousMixin', num_samples: int = 100000) -> float:

        return (self.rvs(num_samples) < other.rvs(num_samples)).mean()

    def __gt__(self, other: 'RVContinuousMixin'):

        return (self.rvs(self._num_samples) > other.rvs(self._num_samples)).mean()

    def __lt__(self, other: 'RVContinuousMixin'):

        return (self.rvs(self._num_samples) < other.rvs(self._num_samples)).mean()
