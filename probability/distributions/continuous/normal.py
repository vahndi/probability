from math import sqrt
from scipy.stats import norm, rv_continuous

from probability.distributions.mixins.rv_continuous_mixin import RVContinuousMixin


class Normal(RVContinuousMixin):

    _parametrization: str

    def __init__(self, mu: float, sigma: float = None, sigma_sq: float = None):

        assert any([sigma, sigma_sq]) and not all([sigma, sigma_sq])

        self._mu: float = mu
        if sigma is not None:
            self._sigma: float = sigma
            self._parametrization: str = 'μσ'
        else:
            self._sigma: float = sqrt(sigma_sq)
            self._parametrization: str = 'μσ²'
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = norm(self._mu, self._sigma)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self._reset_distribution()

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float):
        self._sigma = value
        self._reset_distribution()

    @property
    def sigma_sq(self) -> float:
        return self._sigma ** 2

    @sigma_sq.setter
    def sigma_sq(self, value: float):
        self._sigma = sqrt(value)
        self._reset_distribution()

    def __str__(self):

        if self._parametrization == 'μσ':
            return f'Normal(μ={self._mu}, σ={self._sigma})'
        elif self._parametrization == 'μσ²':
            return f'Normal(μ={self._mu}, σ²={self.sigma_sq})'

    def __repr__(self):

        if self._parametrization == 'μσ':
            return f'Normal(mu={self._mu}, sigma={self._sigma})'
        elif self._parametrization == 'μσ²':
            return f'Normal(mu={self._mu}, sigma_sq={self.sigma_sq})'
