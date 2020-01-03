from probability.distributions import Beta
from probability.distributions.discrete import Binomial
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin


class BetaBinomial(ConjugateMixin):

    def __init__(self, alpha: float = 1, beta: float = 1, n: int = 0, m: int = 0):

        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._m: int = m
        self._calculate_prior()
        self._calculate_likelihood()
        self._calculate_posterior()

    def _calculate_prior(self):

        self._prior: Beta = Beta(alpha=self._alpha, beta=self._beta)

    def _calculate_likelihood(self):

        self._likelihood: Binomial = Binomial(n=self._n, p=self._m / self._n)

    def _calculate_posterior(self):

        self._posterior: Beta = Beta(
            alpha=self._alpha + self._m,
            beta=self._beta + self._n - self._m
        )

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._calculate_prior()

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self._calculate_prior()

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value
        self._calculate_likelihood()
        self._calculate_posterior()

    @property
    def m(self) -> int:
        return self._m

    @m.setter
    def m(self, value: int):
        self._m = value
        self._calculate_likelihood()
        self._calculate_posterior()

    @property
    def prior(self) -> Beta:
        return self._prior

    @property
    def likelihood(self) -> Binomial:
        return self._likelihood

    @property
    def posterior(self) -> Beta:
        return self._posterior

