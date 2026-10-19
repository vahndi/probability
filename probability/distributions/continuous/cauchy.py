from math import pi, tan
from typing import Optional

from scipy.stats import cauchy, rv_continuous

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.mixins.rv_continuous_1d_mixin import (
    RVContinuous1dMixin,
)
from probability.utils import num_format


class Cauchy(
    RVContinuous1dMixin,
    object
):
    """
    The Cauchy distribution (or Lorentz distribution) is a continuous
    probability distribution with heavy tails. It is known for having an
    undefined mean and variance.

    Parameters:
    - x0 (loc): location parameter, specifying the peak / median.
    - gamma (scale): scale parameter, specifying the half-width at half-maximum (gamma > 0).

    https://en.wikipedia.org/wiki/Cauchy_distribution
    """

    def __init__(self, x0: float = 0.0, gamma: float = 1.0):
        """
        Create a new Cauchy distribution.

        :param x0: Location parameter (peak / median).
        :param gamma: Scale parameter (HWHM, gamma > 0).
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be strictly positive, got {gamma}")
        self._x0: float = float(x0)
        self._gamma: float = float(gamma)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = cauchy(
            loc=self._x0,
            scale=self._gamma
        )

    @property
    def x0(self) -> float:
        return self._x0

    @x0.setter
    def x0(self, value: float):
        self._x0 = float(value)
        self._reset_distribution()

    @property
    def loc(self) -> float:
        return self._x0

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        if value <= 0:
            raise ValueError(f"gamma must be strictly positive, got {value}")
        self._gamma = float(value)
        self._reset_distribution()

    @property
    def scale(self) -> float:
        return self._gamma

    @property
    def lower_bound(self) -> float:
        return self.ppf().at(0.01)

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    def mode(self) -> float:
        return self._x0

    def median(self) -> float:
        return self._x0

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.50
    ) -> 'Cauchy':
        """
        Elicit a Cauchy distribution whose central credible interval matches
        [lower, upper] with probability `prob`.

        :param lower: Lower bound of interval.
        :param upper: Upper bound of interval (upper > lower).
        :param prob: Central coverage probability (0 < prob < 1), default 0.50 (interquartile range).
        """
        if not (0.0 < prob < 1.0):
            raise ValueError(f"prob must be between 0 and 1, got {prob}.")
        alpha_tail = (1.0 - prob) / 2.0
        return cls.from_quantiles(
            q1=alpha_tail, v1=lower,
            q2=1.0 - alpha_tail, v2=upper
        )

    @classmethod
    def from_quantiles(
            cls, q1: float, v1: float, q2: float, v2: float
    ) -> 'Cauchy':
        """
        Elicit a Cauchy distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value.
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (v2 > v1).
        """
        if not (0.0 < q1 < q2 < 1.0):
            raise ValueError(f"Probabilities must satisfy 0 < q1 < q2 < 1, got q1={q1}, q2={q2}.")
        if v1 >= v2:
            raise ValueError(f"Quantile values must satisfy v1 < v2, got v1={v1}, v2={v2}.")

        z1 = tan(pi * (q1 - 0.5))
        z2 = tan(pi * (q2 - 0.5))
        gamma = (v2 - v1) / (z2 - z1)
        x0 = v1 - gamma * z1
        return cls(x0=x0, gamma=gamma)

    @staticmethod
    def fit(data: FloatIterable,
            x0: Optional[float] = None,
            gamma: Optional[float] = None) -> 'Cauchy':
        """
        Fit a Cauchy distribution to the data.

        :param data: Iterable of data to fit to.
        :param x0: Optional fixed value for location x0.
        :param gamma: Optional fixed value for scale gamma.
        """
        if x0 is not None and gamma is not None:
            return Cauchy(x0=x0, gamma=gamma)
        kwargs = {}
        if x0 is not None:
            kwargs['floc'] = x0
        if gamma is not None:
            kwargs['fscale'] = gamma
        loc, scale = cauchy.fit(data=data, **kwargs)
        return Cauchy(x0=loc, gamma=scale)

    def __str__(self):
        return (
            f'Cauchy('
            f'x₀={num_format(self._x0, 3)}, '
            f'γ={num_format(self._gamma, 3)})'
        )

    def __repr__(self):
        return f'Cauchy(x0={self._x0}, gamma={self._gamma})'

    def __eq__(self, other: 'Cauchy') -> bool:
        return (
            isinstance(other, Cauchy) and
            abs(self._x0 - other._x0) < 1e-10 and
            abs(self._gamma - other._gamma) < 1e-10
        )

    def __ne__(self, other: 'Cauchy') -> bool:
        return not self.__eq__(other)
