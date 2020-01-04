from scipy.stats import binom, rv_discrete

from probability.distributions.mixins.rv_discrete_mixin import RVDiscreteMixin


class Binomial(RVDiscreteMixin):

    def __init__(self, n: int, p: float):

        self._n: int = n
        self._p: float = p
        self._distribution: rv_discrete = binom(n, p)

    def __str__(self):

        return f'Binomial(n={self._n}, p={self._p})'

    def __repr__(self):

        return f'Binomial(n={self._n}, p={self._p})'
