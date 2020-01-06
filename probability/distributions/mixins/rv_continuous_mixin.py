from matplotlib.axes import Axes
from scipy.stats import rv_continuous
from typing import Iterable

from probability.distributions.mixins.rv_mixins import RVSMixin, MomentMixin, EntropyMixin, MedianMixin, MeanMixin, StDMixin, \
    VarMixin, IntervalMixin, SupportMixin, PDFMixin, CDFMixinC, SFMixinC, PPFMixinC, ISFMixinC


class RVContinuousMixin(
    RVSMixin, MomentMixin, EntropyMixin, MedianMixin, MeanMixin, StDMixin, VarMixin, IntervalMixin, SupportMixin,
    PDFMixin, CDFMixinC, SFMixinC, PPFMixinC, ISFMixinC,
    object
):

    _distribution: rv_continuous
    _num_samples: int = 1000000

    def plot(self, x: Iterable, color: str = 'C0', ax: Axes = None) -> Axes:
        """
        Plot the PDF of the distribution.
        """
        return self.pdf().plot(x=x, color=color, ax=ax)
