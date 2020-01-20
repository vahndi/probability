from math import sqrt
from numpy import array, ndarray
from scipy.stats import norm
from typing import overload, Optional

from probability.custom_types import Array1d
from probability.distributions.continuous.normal import Normal
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin
from probability.utils import none_are_none, all_are_none


class NormalNormal(RVContinuous1dMixin, ConjugateMixin):

    @overload
    def __init__(self, mu_0: float, sigma_sq_0: float, sigma_sq: float, x: Array1d):
        pass

    @overload
    def __init__(self, mu_0: float, tau_0: float, tau: float, x: Array1d):
        pass

    def __init__(self, mu_0: float, x: Array1d,
                 sigma_sq_0: Optional[float] = None, sigma_sq: Optional[float] = None,
                 tau_0: Optional[float] = None, tau: Optional[float] = None):

        assert (
            none_are_none(sigma_sq_0, sigma_sq) and all_are_none(tau_0, tau)
        ) or (
            none_are_none(tau_0, tau) and all_are_none(sigma_sq_0, sigma_sq)
        )

        self._mu_0: float = mu_0
        self._x: ndarray = array(x)
        if sigma_sq_0 is not None:
            self._sigma_sq = sigma_sq
            self._sigma_sq_0 = sigma_sq_0
            self._parametrization = 'μ₀σ₀²σ²'
        else:
            self._sigma_sq = 1 / tau
            self._sigma_sq_0 = 1 / tau_0
            self._parametrization = 'μ₀τ₀τ'

        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution = norm(
            loc=self.mu_0_prime,
            scale=sqrt(self.sigma_sq_0_prime + self.sigma_sq)
        )

    @property
    def mu_0(self) -> float:
        return self._mu_0

    @mu_0.setter
    def mu_0(self, value: float):
        self._mu_0 = value
        self._reset_distribution()

    @property
    def sigma_sq(self) -> float:
        return self._sigma_sq

    @sigma_sq.setter
    def sigma_sq(self, value: float):
        self._sigma_sq = value
        self._reset_distribution()

    @property
    def sigma_sq_0(self) -> float:
        return self._sigma_sq_0

    @sigma_sq_0.setter
    def sigma_sq_0(self, value: float):
        self._sigma_sq_0 = value
        self._reset_distribution()

    @property
    def tau(self) -> float:
        return 1 / self._sigma_sq

    @tau.setter
    def tau(self, value: float):
        self._sigma_sq = 1 / value
        self._reset_distribution()

    @property
    def tau_0(self) -> float:
        return 1 / self._sigma_sq_0

    @tau_0.setter
    def tau_0(self, value: float):
        self._sigma_sq_0 = 1 / value
        self._reset_distribution()

    @property
    def n(self) -> int:
        return len(self._x)

    @property
    def mu_0_prime(self) -> float:
        return (
            (self.tau_0 * self.mu_0 + self.tau * self._x.sum()) /
            (self.tau_0 + self.n * self.tau)
        )

    @property
    def tau_0_prime(self) -> float:
        return self.tau_0 + self.n * self.tau

    @property
    def sigma_sq_0_prime(self) -> float:
        return 1 / self.tau_0_prime

    def prior(self) -> Normal:
        return Normal(
            mu=self.mu_0, sigma_sq=self.sigma_sq_0
        ).with_x_label('μ').prepend_to_label('Prior: ')

    def likelihood(self, **kwargs) -> Normal:
        raise NotImplementedError

    def posterior(self) -> Normal:
        return Normal(
            mu=self.mu_0_prime, sigma_sq=self.sigma_sq_0_prime
        ).with_x_label('μ').prepend_to_label('Posterior: ')

    def __str__(self):

        if self._parametrization == 'μ₀σ₀²σ²':
            return f'NormalNormal(μ₀={self._mu_0}, σ₀²={self._sigma_sq_0}, σ²={self._sigma_sq})'
        elif self._parametrization == 'μ₀τ₀τ':
            return f'NormalNormal(μ₀={self._mu_0}, τ₀={self.tau_0}, τ={self.tau})'

    def __repr__(self):

        if self._parametrization == 'μ₀σ₀²σ²':
            return f'NormalNormal(mu_0={self._mu_0}, sigma_sq_0={self._sigma_sq_0}, sigma_sq={self._sigma_sq})'
        elif self._parametrization == 'μ₀τ₀τ':
            return f'NormalNormal(mu_0={self._mu_0}, tau_0={self.tau_0}, tau={self.tau})'
