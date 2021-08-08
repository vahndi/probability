from typing import Optional

from scipy.stats import rv_continuous, gamma

from compound_types.built_ins import FloatIterable
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

    def __init__(self, alpha: float, beta: float,
                 parameterization: str = 'αβ'):
        """
        Create a new gamma distribution.

        :param alpha: Shape parameter: Interpretation can be number of events
                      that we are waiting on to happen. When alpha = 1 we get
                      the exponential distribution.
        :param beta: Rate parameter: Interpretation can be the rate at which
                     events happen.
        :param parameterization: One of ['αβ', 'kθ']
        """
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

        return Gamma(alpha=alpha, beta=beta,
                     parameterization='αβ')

    @staticmethod
    def from_k_theta(k: float, theta: float) -> 'Gamma':

        return Gamma(alpha=k, beta=1 / theta,
                     parameterization='kθ')

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

    def mode(self) -> float:
        if self.alpha >= 1:
            return (self.alpha - 1) / self.beta
        else:
            raise ValueError("Can't calculate mode for this distribution.")

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    @staticmethod
    def fit(data: FloatIterable,
            alpha: Optional[float] = None,
            beta: Optional[float] = None) -> 'Gamma':
        """
        Fit a Gamma distribution to the data.

        :param data: Iterable of data to fit to.
        :param alpha: Optional fixed value for alpha.
        :param beta: Optional fixed value for beta.
        """
        kwargs = {}
        if alpha is not None:
            kwargs['fa'] = alpha
        if beta is not None:
            kwargs['fscale'] = 1 / beta
        alpha, loc, scale = gamma.fit(data=data, floc=0, **kwargs)
        return Gamma(alpha=alpha, beta=1 / scale)

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

    def __eq__(self, other: 'Gamma') -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )

    def __ne__(self, other: 'Gamma') -> bool:

        return not self.__eq__(other)
