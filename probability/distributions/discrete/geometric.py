from scipy.stats import geom, rv_discrete

from probability.distributions.mixins.attributes import PFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin


class Geometric(
    RVDiscrete1dMixin,
    PFloatDMixin,
    CalculableMixin,
    object
):
    """
    The (shifted) geometric distribution gives the probability that the first
    occurrence of success requires k independent trials, each with success
    probability p.

    https://en.wikipedia.org/wiki/Geometric_distribution
    """
    def __init__(self, p: float):
        """
        Create a new shifted geometric distribution.

        :param p: Probability of success in any individual trial.
        """
        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = geom(self._p)

    def __str__(self):

        return f'Geometric(p={self._p: 0.2f})'

    def __repr__(self):

        return f'Geometric(p={self._p})'
