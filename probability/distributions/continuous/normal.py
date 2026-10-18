from math import sqrt
from typing import overload, Optional, Union

from scipy.stats import norm, rv_continuous

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.elicitation import elicit_normal
from probability.distributions.mixins.attributes import MuFloatDMixin, \
    SigmaFloatDMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import (
    any_are_not_none, any_are_none, is_scalar, num_format
)


class Normal(
    RVContinuous1dMixin,
    MuFloatDMixin,
    SigmaFloatDMixin,
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

    def _new_like(self, mu: float, sigma: float,
                  other: Optional['Normal'] = None) -> 'Normal':
        use_sq = (
            self._parameterization == 'μσ²' and
            (other is None or other._parameterization == 'μσ²')
        )
        if use_sq:
            return Normal(mu=mu, sigma_sq=sigma ** 2)
        return Normal(mu=mu, sigma=sigma)

    def __neg__(self) -> 'Normal':
        return self._new_like(mu=-self._mu, sigma=self._sigma)

    def __add__(self, other):
        if isinstance(other, Normal):
            return self._new_like(
                mu=self._mu + other._mu,
                sigma=sqrt(self._sigma ** 2 + other._sigma ** 2),
                other=other
            )
        elif is_scalar(other):
            return self._new_like(
                mu=self._mu + float(other),
                sigma=self._sigma
            )
        return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        elif is_scalar(other):
            return self._new_like(
                mu=float(other) + self._mu,
                sigma=self._sigma
            )
        return super().__radd__(other)

    def __sub__(self, other):
        if isinstance(other, Normal):
            return self._new_like(
                mu=self._mu - other._mu,
                sigma=sqrt(self._sigma ** 2 + other._sigma ** 2),
                other=other
            )
        elif is_scalar(other):
            return self._new_like(
                mu=self._mu - float(other),
                sigma=self._sigma
            )
        return super().__sub__(other)

    def __rsub__(self, other):
        if is_scalar(other):
            return self._new_like(
                mu=float(other) - self._mu,
                sigma=self._sigma
            )
        return super().__rsub__(other)

    def __mul__(self, other):
        if is_scalar(other):
            c = float(other)
            if c != 0:
                return self._new_like(
                    mu=self._mu * c,
                    sigma=abs(c) * self._sigma
                )
        return super().__mul__(other)

    def __rmul__(self, other):
        if is_scalar(other):
            return self.__mul__(other)
        return super().__rmul__(other)

    def __truediv__(self, other):
        if is_scalar(other):
            c = float(other)
            if c == 0:
                raise ZeroDivisionError('Cannot divide Normal distribution by 0')
            return self._new_like(
                mu=self._mu / c,
                sigma=self._sigma / abs(c)
            )
        return super().__truediv__(other)

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'Normal':
        """
        Elicit a Normal distribution whose central credible interval matches
        [lower, upper] with probability `prob`.

        :param lower: Lower bound of interval.
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
    ) -> 'Normal':
        """
        Elicit a Normal distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value.
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (v2 > v1).
        """
        mu, sigma = elicit_normal(q1=q1, v1=v1, q2=q2, v2=v2)
        return cls(mu=mu, sigma=sigma)

    @classmethod
    def from_mean_and_std(cls, mean: float, std: float) -> 'Normal':
        """
        Construct a Normal distribution matching a given mean and standard deviation.

        :param mean: Arithmetic mean mu.
        :param std: Standard deviation sigma (std > 0).
        """
        if std <= 0.0:
            raise ValueError(f"std must be strictly positive, got {std}.")
        return cls(mu=mean, sigma=std)

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
            isinstance(other, Normal) and
            abs(self._mu - other._mu) < 1e-10 and
            abs(self._sigma - other._sigma) < 1e-10
        )

    def __ne__(self, other: 'Normal') -> bool:

        return not self.__eq__(other)

