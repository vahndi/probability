from math import sqrt
from typing import overload, Optional

from scipy.stats import norm, rv_continuous

from probability.distributions.mixins.attributes import MuFloatDMixin, \
    SigmaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import any_are_not_none, any_are_none, num_format


class Normal(
    RVContinuous1dMixin,
    MuFloatDMixin,
    SigmaFloatDMixin,
    CalculableMixin,
    object
):
    """
    The normal distribution is a type of continuous probability distribution for
    a real-valued random variable.

    https://en.wikipedia.org/wiki/Normal_distribution
    """
    _parameterization: str

    @overload
    def __init__(self, mu: float, sigma: float):
        pass

    @overload
    def __init__(self, mu: float, sigma_sq: float):
        pass

    def __init__(self, mu: float,
                 sigma: Optional[float] = None,
                 sigma_sq: Optional[float] = None):

        assert (
            any_are_not_none(sigma, sigma_sq) and
            any_are_none(sigma, sigma_sq)
        )

        self._mu: float = mu
        if sigma is not None:
            self._sigma: float = sigma
            self._parameterization: str = 'μσ'
        else:
            self._sigma: float = sqrt(sigma_sq)
            self._parameterization: str = 'μσ²'
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = norm(self._mu, self._sigma)

    @staticmethod
    def StandardNormal():
        return Normal(mu=0, sigma=1)

    @property
    def sigma_sq(self) -> float:
        return self._sigma ** 2

    @sigma_sq.setter
    def sigma_sq(self, value: float):
        self._sigma = sqrt(value)
        self._reset_distribution()

    def __str__(self):

        if self._parameterization == 'μσ':
            return (
                f'Normal('
                f'μ={num_format(self._mu, 3)}, '
                f'σ={num_format(self._sigma, 3)})'
            )
        elif self._parameterization == 'μσ²':
            return (
                f'Normal('
                f'μ={num_format(self._mu, 3)}, '
                f'σ²={num_format(self.sigma_sq, 3)})'
            )

    def __repr__(self):

        if self._parameterization == 'μσ':
            return f'Normal(mu={self._mu}, sigma={self._sigma})'
        elif self._parameterization == 'μσ²':
            return f'Normal(mu={self._mu}, sigma_sq={self.sigma_sq})'
