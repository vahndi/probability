from scipy.stats import binom, rv_discrete

from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class Binomial(RVDiscrete1dMixin):

    def __init__(self, n: int, p: float):

        self._n: int = n
        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = binom(self._n, self._p)

    @property
    def n(self) -> float:
        return self._n

    @n.setter
    def n(self, value: float):
        self._n = value
        self._reset_distribution()

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float):
        self._p = value
        self._reset_distribution()

    def __str__(self):

        return f'Binomial(n={self._n}, p={self._p})'

    def __repr__(self):

        return f'Binomial(n={self._n}, p={self._p})'
