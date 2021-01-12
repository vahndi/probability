from math import sqrt

from numpy import array, ndarray, power
from scipy.stats import t, rv_continuous

from probability.custom_types.external_custom_types import Array1d
from probability.custom_types.compound_types import RVMixin
from probability.distributions.continuous.inverse_gamma import InverseGamma
from probability.distributions.mixins.conjugate import ConjugateMixin
from probability.utils import num_format


class _InvGammaNormalConjugate(ConjugateMixin, object):

    def __init__(self, alpha: float, beta: float,
                 x: Array1d, mu: float):

        self._alpha: float = alpha
        self._beta: float = beta
        self._x: ndarray = array(x)
        self._mu = mu
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = t(
            2 * self.alpha_prime,
            loc=self._mu,
            scale=sqrt(self.beta_prime / self.alpha_prime)
        )

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self.n / 2

    @property
    def beta_prime(self) -> float:
        return self._beta + (self.n / 2) * power(self._x - self._mu, 2).mean()

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
    def x(self) -> ndarray:
        return self._x

    @x.setter
    def x(self, value: Array1d):
        self._x = value
        self._reset_distribution()

    @property
    def n(self) -> int:
        return len(self._x)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self._reset_distribution()

    def prior(self) -> InverseGamma:

        return InverseGamma(
            alpha=self._alpha, beta=self._beta
        ).with_x_label('σ²').prepend_to_label('Prior: ')

    def likelihood(self, **kwargs) -> RVMixin:
        pass

    def posterior(self) -> InverseGamma:

        return InverseGamma(
            alpha=self.alpha_prime, beta=self.beta_prime
        ).with_x_label('σ²').prepend_to_label('Posterior: ')

    def __str__(self):

        return (
            f'InvGammaNormalConjugate('
            f'α={num_format(self._alpha, 3)}, '
            f'β={num_format(self._beta, 3)}, '
            f'μ={num_format(self._mu, 3)}, '
            f'n={self.n})'
        )

    def __repr__(self):

        return (
            f'InvGammaNormalConjugate('
            f'alpha={self._alpha}, '
            f'beta={self._beta}, '
            f'mu={self._mu}, '
            f'n={self.n})'
        )
