from math import exp, log
from typing import Optional

from scipy.stats import weibull_min, rv_continuous

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.mixins.rv_continuous_1d_mixin import (
    RVContinuous1dMixin,
)
from probability.utils import num_format


class Weibull(
    RVContinuous1dMixin,
    object
):
    """
    The Weibull distribution is a continuous probability distribution widely
    used in reliability engineering, survival analysis, and extreme value theory.

    Parameters:
    - k (shape): shape parameter (k > 0). When k=1, Weibull becomes Exponential.
    - lambda_ (scale): scale parameter (lambda_ > 0).

    https://en.wikipedia.org/wiki/Weibull_distribution
    """

    def __init__(self, k: float, lambda_: float = 1.0):
        """
        Create a new Weibull distribution.

        :param k: Shape parameter (k > 0).
        :param lambda_: Scale parameter (lambda_ > 0).
        """
        if k <= 0:
            raise ValueError(f"k must be strictly positive, got {k}")
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be strictly positive, got {lambda_}")

        self._k: float = float(k)
        self._lambda: float = float(lambda_)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = weibull_min(
            c=self._k,
            scale=self._lambda
        )

    @property
    def k(self) -> float:
        return self._k

    @k.setter
    def k(self, value: float):
        if value <= 0:
            raise ValueError(f"k must be strictly positive, got {value}")
        self._k = float(value)
        self._reset_distribution()

    @property
    def shape(self) -> float:
        return self._k

    @property
    def lambda_(self) -> float:
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: float):
        if value <= 0:
            raise ValueError(f"lambda_ must be strictly positive, got {value}")
        self._lambda = float(value)
        self._reset_distribution()

    @property
    def scale(self) -> float:
        return self._lambda

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    def mode(self) -> float:
        if self._k > 1.0:
            return self._lambda * (((self._k - 1.0) / self._k) ** (1.0 / self._k))
        else:
            return 0.0

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'Weibull':
        """
        Elicit a Weibull distribution whose central credible interval matches
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
    ) -> 'Weibull':
        """
        Elicit a Weibull distribution matching two arbitrary cumulative quantiles:
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

        y1 = log(-log(1.0 - q1))
        y2 = log(-log(1.0 - q2))
        k = (y2 - y1) / (log(v2) - log(v1))
        log_lambda = log(v1) - (y1 / k)
        lambda_ = exp(log_lambda)
        return cls(k=k, lambda_=lambda_)

    @staticmethod
    def fit(data: FloatIterable,
            k: Optional[float] = None,
            lambda_: Optional[float] = None) -> 'Weibull':
        """
        Fit a Weibull distribution to the data.

        :param data: Iterable of positive values to fit to.
        :param k: Optional fixed value for shape k.
        :param lambda_: Optional fixed value for scale lambda_.
        """
        if k is not None and lambda_ is not None:
            return Weibull(k=k, lambda_=lambda_)
        kwargs = {}
        if k is not None:
            kwargs['f0'] = k
        if lambda_ is not None:
            kwargs['fscale'] = lambda_
        c, loc, scale = weibull_min.fit(data=data, floc=0, **kwargs)
        return Weibull(k=float(c), lambda_=float(scale))

    def __str__(self):
        return (
            f'Weibull('
            f'k={num_format(self._k, 3)}, '
            f'λ={num_format(self._lambda, 3)})'
        )

    def __repr__(self):
        return f'Weibull(k={self._k}, lambda_={self._lambda})'

    def __eq__(self, other: 'Weibull') -> bool:
        return (
            isinstance(other, Weibull) and
            abs(self._k - other._k) < 1e-10 and
            abs(self._lambda - other._lambda) < 1e-10
        )

    def __ne__(self, other: 'Weibull') -> bool:
        return not self.__eq__(other)
