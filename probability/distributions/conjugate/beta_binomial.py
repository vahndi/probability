from numpy import ndarray
from scipy.stats import betabinom
from typing import Union, Iterable

from probability.distributions import Beta
from probability.distributions.discrete import Binomial
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class BetaBinomial(RVDiscrete1dMixin, ConjugateMixin):

    def __init__(self, n: int, alpha: float = 1, beta: float = 1):
        """
        :param n: Number of trials.
        :param alpha: Value for the α hyper-parameter of the prior Beta distribution.
        :param beta: Value for the α hyper-parameter of the prior Beta distribution.
        """
        self._n: int = n
        self._alpha: float = alpha
        self._beta: float = beta
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution = betabinom(self._n, self._alpha, self._beta)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self._reset_distribution()

    @property
    def n(self) -> float:
        return self._n

    @n.setter
    def n(self, value: float):
        self._n = value
        self._reset_distribution()

    def prior(self) -> Beta:
        return Beta(alpha=self._alpha, beta=self._beta)

    def likelihood(self, m: float) -> Binomial:
        return Binomial(n=self._n, p=m / self._n)  # * comb(n, k)

    def posterior(self, m: int) -> Beta:
        return Beta(alpha=self._alpha + m, beta=self._beta + self._n - m)

    def predict_proba(self, m: Union[ndarray, Iterable, float]):

        return self._distribution.pmf(k=m)

    def __str__(self):

        return f'BetaBinomial(n={self._n}, α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'BetaBinomial(n={self._n}, alpha={self._alpha}, beta={self._beta})'

