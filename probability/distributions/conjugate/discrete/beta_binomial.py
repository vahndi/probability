from scipy.stats import betabinom

from probability.custom_types import RVMixin
from probability.distributions import Beta
from probability.distributions.discrete import Binomial
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class BetaBinomial(RVDiscrete1dMixin, ConjugateMixin):

    def __init__(self, n: int, alpha: float = 1, beta: float = 1):

        self._n: int = n
        self._alpha: float = alpha
        self._beta: float = beta
        self._distribution = betabinom(n, alpha, beta)

    def __str__(self):

        return f'BetaBinomial(n={self._n}, α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'BetaBinomial(n={self._n}, alpha={self._alpha}, beta={self._beta})'

    def prior(self) -> Beta:
        return Beta(alpha=self._alpha, beta=self._beta)

    def likelihood(self, m: float) -> Binomial:
        return Binomial(n=self._n, p=m / self._n)

    def posterior(self, m: int) -> Beta:
        return Beta(alpha=self._alpha + m, beta=self._beta + self._n - m)
