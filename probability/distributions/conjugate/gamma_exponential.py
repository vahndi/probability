from scipy.stats import lomax

from probability.distributions.continuous.gamma import Gamma
from probability.distributions.continuous.exponential import Exponential
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class GammaExponential(RVContinuous1dMixin, ConjugateMixin):
    """
    Class for calculating Bayesian probabilities using the Gamma-Exponential distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` are the hyper-parameters of the Gamma prior.
    * `α > 0`
    * `β > 0`
    * Interpretation is α observations that sum to β.

    Posterior Hyper-parameters
    --------------------------
    * `n` is the number of observations.
    * `x_mean` is the average value (e.g. duration) of x over `n` observations.

    Model parameters
    ----------------
    * `P(x)` is the probability of observing an event at a rate of `x`.
    * `0 ≤ x`

    Links
    -----
    * https://en.wikipedia.org/wiki/Gamma_distribution
    * https://en.wikipedia.org/wiki/Exponential_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_discrete_distribution
    """
    def __init__(self, alpha: float, beta: float, n: int, x_mean: float):
        """
        :param alpha: Value for the α hyper-parameter of the prior Gamma distribution.
        :param beta: Value for the β hyper-parameter of the prior Gamma distribution.
        :param n: Number of observations.
        :param x_mean: Average duration of, or time between, observations.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._x_mean: float = x_mean
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution = lomax(c=self.alpha_prime, scale=self.beta_prime)

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
    def n(self):
        return self._n

    @n.setter
    def n(self, value: float):
        self._n = value
        self._reset_distribution()

    @property
    def x_mean(self) -> float:
        return self._x_mean

    @x_mean.setter
    def x_mean(self, value: float):
        self._x_mean = value
        self._reset_distribution()

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._n

    @property
    def beta_prime(self) -> float:
        return self._beta + self._n * self._x_mean

    def prior(self) -> Gamma:
        return Gamma(
            alpha=self._alpha, beta=self._beta
        ).with_x_label('λ').prepend_to_label('Prior: ')

    def likelihood(self, x: float) -> Exponential:
        raise NotImplementedError

    def posterior(self) -> Gamma:

        return Gamma(
            alpha=self.alpha_prime, beta=self.beta_prime
        ).with_x_label('λ').prepend_to_label('Posterior: ')

    def __str__(self):

        return f'GammaExponential(α={self._alpha}, β={self._beta}, n={self._n}, x̄={self._x_mean})'

    def __repr__(self):

        return f'GammaExponential(alpha={self._alpha}, beta={self._beta}, n={self._n}, x_mean={self._x_mean})'
