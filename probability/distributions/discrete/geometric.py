from scipy.stats import geom, rv_discrete

from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class Geometric(RVDiscrete1dMixin):
    """
    Class to represent a (shifted) geometric distribution.
    """
    def __init__(self, p: float):

        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = geom(self._p)

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float):
        self._p = value
        self._reset_distribution()

    def __str__(self):

        return f'Geometric(p={self._p})'

    def __repr__(self):

        return f'Geometric(p={self._p})'
