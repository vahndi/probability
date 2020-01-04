from scipy.stats import betabinom

from probability.distributions.mixins.rv_discrete_mixin import RVDiscreteMixin


class BetaBinomial(RVDiscreteMixin):

    def __init__(self, n: int, alpha: float = 1, beta: float = 1):

        self._n: int = n
        self._alpha: float = alpha
        self._beta: float = beta
        self._distribution = betabinom(n, alpha, beta)

    def __str__(self):

        return f'BetaBinomial(n={self._n}, α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'BetaBinomial(n={self._n}, alpha={self._alpha}, beta={self._beta})'
