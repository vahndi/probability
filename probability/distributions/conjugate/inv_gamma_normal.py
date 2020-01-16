from numpy import array, ndarray, power
from scipy.stats import invgamma, rv_continuous

from probability.custom_types import RVMixin, Array1d
from probability.distributions.continuous.inv_gamma import InvGamma
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class InvGammaNormal(RVContinuous1dMixin, ConjugateMixin):

    def __init__(self, alpha: float, beta: float,
                 x: Array1d, mu: float):

        self._alpha: float = alpha
        self._beta: float = beta
        self._x: ndarray = array(x)
        self._mu = mu
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = invgamma(a=self.alpha_prime, scale=self.beta_prime)

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

    def prior(self, **kwargs) -> InvGamma:

        return InvGamma(alpha=self._alpha, beta=self._beta).with_x_label('σ²')

    def likelihood(self, **kwargs) -> RVMixin:
        pass

    def posterior(self, **kwargs) -> InvGamma:

        return InvGamma(alpha=self.alpha_prime, beta=self.beta_prime).with_x_label('σ²')
