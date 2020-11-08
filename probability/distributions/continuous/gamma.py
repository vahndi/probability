from scipy.stats import rv_continuous, gamma

from probability.distributions.mixins.attributes import AlphaFloatDMixin, \
    BetaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class Gamma(
    RVContinuous1dMixin,
    AlphaFloatDMixin,
    BetaFloatDMixin,
    CalculableMixin,
    object
):
    """
    The gamma distribution is a two-parameter family of continuous probability
    distributions. The exponential distribution, Erlang distribution, and
    chi-squared distribution are special cases of the gamma distribution.

    The parameterization with k and θ appears to be more common in econometrics
    and certain other applied fields, where for example the gamma distribution
    is frequently used to model waiting times.

    The parameterization with α and β is more common in Bayesian statistics,
    where the gamma distribution is used as a conjugate prior distribution for
    various types of inverse scale (rate) parameters, such as the λ (rate) of
    an exponential distribution or of a Poisson distribution

    https://en.wikipedia.org/wiki/Gamma_distribution
    """

    _parameterization: str

    def __init__(self, alpha: float, beta: float, parameterization: str = 'αβ'):

        assert parameterization in ('αβ', 'kθ')
        self._alpha: float = alpha
        self._beta: float = beta
        self._parameterization: str = parameterization
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = gamma(
            a=self._alpha, scale=1 / self._beta
        )

    @staticmethod
    def from_alpha_beta(alpha: float, beta: float) -> 'Gamma':

        return Gamma(alpha=alpha, beta=beta, parameterization='αβ')

    @staticmethod
    def from_k_theta(k: float, theta: float) -> 'Gamma':

        return Gamma(alpha=k, beta=1 / theta, parameterization='kθ')

    @property
    def k(self) -> float:
        return self._alpha

    @k.setter
    def k(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def theta(self):
        return 1 / self._beta

    @theta.setter
    def theta(self, value: float):
        self._beta = 1 / value
        self._reset_distribution()

    def __str__(self):

        if self._parameterization == 'αβ':
            return (
                f'Gamma('
                f'α={num_format(self._alpha, 3)}, '
                f'β={num_format(self._beta, 3)})'
            )
        elif self._parameterization == 'kθ':
            return (
                f'Gamma('
                f'k={num_format(self._alpha, 3)}, '
                f'θ={num_format(1 / self._beta, 3)})'
            )

    def __repr__(self):

        if self._parameterization == 'αβ':
            return f'Gamma(alpha={self._alpha}, beta={self._beta})'
        elif self._parameterization == 'kθ':
            return f'Gamma(k={self._alpha}, theta={1 / self._beta})'
