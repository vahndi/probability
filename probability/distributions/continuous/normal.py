from math import sqrt
from typing import overload, Optional

from scipy.stats import norm, rv_continuous

from compound_types.built_ins import FloatIterable
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

    @property
    def lower_bound(self) -> float:
        return self.isf().at(0.99)

    @property
    def upper_bound(self) -> float:
        return self.isf().at(0.01)

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

    def mode(self) -> float:

        return self._mu

    @staticmethod
    def fit(data: FloatIterable,
            mu: Optional[float] = None,
            sigma: Optional[float] = None) -> 'Normal':
        """
        Fit a Normal distribution to the data.

        :param data: Iterable of data to fit to.
        :param mu: Optional fixed value for mu.
        :param sigma: Optional fixed value for sigma.
        """
        kwargs = {}
        for arg, kw in zip(
            (mu, sigma),
            ('floc', 'fscale')
        ):
            if arg is not None:
                kwargs[kw] = arg
        loc, scale = norm.fit(data=data, **kwargs)
        return Normal(mu=loc, sigma=scale)

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

    def __eq__(self, other: 'Normal') -> bool:

        return (
            abs(self._mu - other._mu) < 1e-10 and
            abs(self._sigma - other._sigma) < 1e-10
        )

    def __ne__(self, other: 'Normal') -> bool:

        return not self.__eq__(other)
