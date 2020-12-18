from scipy.stats import laplace, rv_continuous

from probability.distributions.mixins.attributes import MuFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class Laplace(
    RVContinuous1dMixin,
    MuFloatDMixin,
    CalculableMixin,
    object
):
    """
    The Laplace distribution is also sometimes called the double exponential
    distribution, because it can be thought of as two exponential distributions
    (with an additional location parameter) spliced together back-to-back.
    The difference between two independent identically distributed exponential
    random variables is governed by a Laplace distribution

    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    def __init__(self, mu: float, b: float):
        self._mu: float = mu
        self._b: float = b
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = laplace(self._mu, self._b)

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        self._b = value
        self._reset_distribution()

    def mode(self) -> float:

        return self._mu

    def __str__(self):
        return (
            f'Laplace('
            f'μ={num_format(self._mu, 3)}, '
            f'b={num_format(self._b, 3)})'
        )

    def __repr__(self):
        return f'Laplace(mu={self._mu}, b={self._b})'

    def __eq__(self, other: 'Laplace') -> bool:

        return (
            abs(self._mu - other._mu) < 1e-10 and
            abs(self._b - other._b) < 1e-10
        )
