from math import exp, log
from typing import Optional

from scipy.stats import pareto, rv_continuous

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.mixins.rv_continuous_1d_mixin import (
    RVContinuous1dMixin,
)
from probability.utils import num_format


class Pareto(
    RVContinuous1dMixin,
    object
):
    """
    The Pareto distribution (Type I) is a power-law probability distribution
    that is used in description of social, quality control, scientific,
    geophysical, actuarial, and many other types of observable phenomena
    (e.g., 80-20 rule, wealth distribution, city sizes).

    Parameters:
    - alpha (shape): tail index / shape parameter (alpha > 0).
    - x_m (scale): minimum possible value / scale parameter (x_m > 0).

    https://en.wikipedia.org/wiki/Pareto_distribution
    """

    def __init__(self, alpha: float, x_m: float = 1.0):
        """
        Create a new Pareto distribution.

        :param alpha: Shape parameter (alpha > 0).
        :param x_m: Minimum value / scale parameter (x_m > 0).
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be strictly positive, got {alpha}")
        if x_m <= 0:
            raise ValueError(f"x_m must be strictly positive, got {x_m}")

        self._alpha: float = float(alpha)
        self._x_m: float = float(x_m)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = pareto(
            b=self._alpha,
            scale=self._x_m
        )

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value <= 0:
            raise ValueError(f"alpha must be strictly positive, got {value}")
        self._alpha = float(value)
        self._reset_distribution()

    @property
    def shape(self) -> float:
        return self._alpha

    @property
    def x_m(self) -> float:
        return self._x_m

    @x_m.setter
    def x_m(self, value: float):
        if value <= 0:
            raise ValueError(f"x_m must be strictly positive, got {value}")
        self._x_m = float(value)
        self._reset_distribution()

    @property
    def scale(self) -> float:
        return self._x_m

    @property
    def lower_bound(self) -> float:
        return self._x_m

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    def mode(self) -> float:
        return self._x_m

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'Pareto':
        """
        Elicit a Pareto distribution whose central credible interval matches
        [lower, upper] with probability `prob`.

        :param lower: Lower bound of interval (lower > 0).
        :param upper: Upper bound of interval (upper > lower).
        :param prob: Central coverage probability (0 < prob < 1), default 0.95.
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
    ) -> 'Pareto':
        """
        Elicit a Pareto distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value (v1 > 0).
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (v2 > v1).
        """
        if not (0.0 < q1 < q2 < 1.0):
            raise ValueError(f"Probabilities must satisfy 0 < q1 < q2 < 1, got q1={q1}, q2={q2}.")
        if v1 <= 0.0 or v2 <= 0.0:
            raise ValueError(f"Quantile values must be strictly positive, got v1={v1}, v2={v2}.")
        if v1 >= v2:
            raise ValueError(f"Quantile values must satisfy v1 < v2, got v1={v1}, v2={v2}.")

        alpha = (log(1.0 - q1) - log(1.0 - q2)) / (log(v2) - log(v1))
        log_xm = log(v1) + log(1.0 - q1) / alpha
        x_m = exp(log_xm)
        return cls(alpha=alpha, x_m=x_m)

    @staticmethod
    def fit(data: FloatIterable,
            alpha: Optional[float] = None,
            x_m: Optional[float] = None) -> 'Pareto':
        """
        Fit a Pareto distribution to the data.

        :param data: Iterable of positive values to fit to.
        :param alpha: Optional fixed value for shape alpha.
        :param x_m: Optional fixed value for scale x_m.
        """
        if alpha is not None and x_m is not None:
            return Pareto(alpha=alpha, x_m=x_m)
        kwargs = {}
        if alpha is not None:
            kwargs['f0'] = alpha
        if x_m is not None:
            kwargs['fscale'] = x_m
        b, loc, scale = pareto.fit(data=data, floc=0, **kwargs)
        return Pareto(alpha=float(b), x_m=float(scale))

    def __str__(self):
        return (
            f'Pareto('
            f'α={num_format(self._alpha, 3)}, '
            f'xₘ={num_format(self._x_m, 3)})'
        )

    def __repr__(self):
        return f'Pareto(alpha={self._alpha}, x_m={self._x_m})'

    def __eq__(self, other: 'Pareto') -> bool:
        return (
            isinstance(other, Pareto) and
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._x_m - other._x_m) < 1e-10
        )

    def __ne__(self, other: 'Pareto') -> bool:
        return not self.__eq__(other)
