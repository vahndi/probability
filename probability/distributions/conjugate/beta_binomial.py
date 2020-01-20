from typing import Optional

from scipy.stats import betabinom

from probability.distributions.continuous.beta import Beta
from probability.distributions.discrete import Binomial
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class BetaBinomial(RVDiscrete1dMixin, ConjugateMixin):
    """
    Class for calculating Bayesian probabilities using the Beta-Binomial distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` are the hyper-parameters of the Beta prior.
    * `α > 0`
    * `β > 0`
    * Interpretation is α-1 successes and β-1 failures.

    Posterior Hyper-parameters
    --------------------------
    * `n` is the number of trials.
    * `m` is the number of successes over `n` trials.

    Model parameters
    ----------------
    * `P(x=1)`, or `θ`, is the probability of a successful trial.
    * `0 ≤ θ ≤ 1`

    Links
    -----
    * https://en.wikipedia.org/wiki/Beta_distribution
    * https://en.wikipedia.org/wiki/Binomial_distribution
    * https://en.wikipedia.org/wiki/Beta-binomial_distribution
    https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
    """
    def __init__(self, alpha: float, beta: float, n: int, m: int):
        """
        :param alpha: Value for the α hyper-parameter of the prior Beta distribution.
        :param beta: Value for the β hyper-parameter of the prior Beta distribution.
        :param n: Number of trials.
        :param m: Number of successes.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._m: int = m
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

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, value: float):
        self._m = value

    def prior(self) -> Beta:
        return Beta(
            alpha=self._alpha, beta=self._beta
        ).with_x_label('θ').prepend_to_label('Prior: ')

    def likelihood(self, m: Optional[int] = None) -> Binomial:
        m = m if m is not None else self._m
        return Binomial(n=self._n, p=m / self._n).with_x_label('k')  # * comb(n, k)

    def posterior(self, m: Optional[int] = None) -> Beta:
        m = m if m is not None else self._m
        return Beta(
            alpha=self._alpha + m, beta=self._beta + self._n - m
        ).with_x_label('θ').prepend_to_label('Posterior: ')

    def __str__(self):

        return f'BetaBinomial(n={self._n}, α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'BetaBinomial(n={self._n}, alpha={self._alpha}, beta={self._beta})'
