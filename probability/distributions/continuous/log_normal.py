from math import exp, sqrt
from typing import Optional

import numpy as np
from scipy.stats import lognorm, rv_continuous

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.elicitation import elicit_log_normal
from probability.distributions.mixins.attributes import (
    MuFloatDMixin,
    SigmaFloatDMixin,
)
from probability.distributions.mixins.rv_continuous_1d_mixin import (
    RVContinuous1dMixin,
)
from probability.utils import num_format


class LogNormal(
    RVContinuous1dMixin,
    MuFloatDMixin,
    SigmaFloatDMixin,
    object
):
    """
    The log-normal (or Galton) distribution is a continuous probability
    distribution of a random variable whose logarithm is normally distributed.
    Thus, if the random variable X is log-normally distributed, then
    Y = ln(X) has a normal distribution.

    Parameters mu and sigma are the mean and standard deviation of ln(X).

    https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    def __init__(self, mu: float, sigma: float):
        """
        Create a new LogNormal distribution.

        :param mu: Location parameter (mean of ln(X)).
        :param sigma: Scale/shape parameter (standard deviation of ln(X), sigma > 0).
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        self._mu: float = float(mu)
        self._sigma: float = float(sigma)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = lognorm(
            s=self._sigma,
            scale=exp(self._mu)
        )

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    def mode(self) -> float:
        return exp(self._mu - (self._sigma ** 2))

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'LogNormal':
        """
        Elicit a LogNormal distribution whose central credible interval matches
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
    ) -> 'LogNormal':
        """
        Elicit a LogNormal distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value (v1 > 0).
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (v2 > v1).
        """
        mu, sigma = elicit_log_normal(q1=q1, v1=v1, q2=q2, v2=v2)
        return cls(mu=mu, sigma=sigma)

    @classmethod
    def from_mean_and_std(cls, mean: float, std: float) -> 'LogNormal':
        """
        Construct a LogNormal distribution matching a given mean and standard
        deviation in the natural (untransformed) domain.

        :param mean: Arithmetic mean E[X] (mean > 0).
        :param std: Standard deviation SD[X] (std > 0).
        """
        if mean <= 0.0:
            raise ValueError(f"mean must be strictly positive, got {mean}.")
        if std <= 0.0:
            raise ValueError(f"std must be strictly positive, got {std}.")

        var = std ** 2
        sigma_sq = float(np.log(1.0 + var / (mean ** 2)))
        sigma = sqrt(sigma_sq)
        mu = float(np.log(mean) - 0.5 * sigma_sq)
        return cls(mu=mu, sigma=sigma)

    @staticmethod
    def fit(data: FloatIterable,
            mu: Optional[float] = None,
            sigma: Optional[float] = None) -> 'LogNormal':
        """
        Fit a LogNormal distribution to the data.

        :param data: Iterable of positive values to fit to.
        :param mu: Optional fixed value for mu (log-scale mean).
        :param sigma: Optional fixed value for sigma (log-scale std).
        """
        if mu is not None and sigma is not None:
            return LogNormal(mu=mu, sigma=sigma)
        kwargs = {}
        if mu is not None:
            kwargs['fscale'] = exp(mu)
        if sigma is not None:
            kwargs['f0'] = sigma
        s, loc, scale = lognorm.fit(data=data, floc=0, **kwargs)
        return LogNormal(mu=float(np.log(scale)), sigma=float(s))

    def __str__(self):
        return (
            f'LogNormal('
            f'μ={num_format(self._mu, 3)}, '
            f'σ={num_format(self._sigma, 3)})'
        )

    def __repr__(self):
        return f'LogNormal(mu={self._mu}, sigma={self._sigma})'

    def __eq__(self, other: 'LogNormal') -> bool:
        return (
            isinstance(other, LogNormal) and
            abs(self._mu - other._mu) < 1e-10 and
            abs(self._sigma - other._sigma) < 1e-10
        )

    def __ne__(self, other: 'LogNormal') -> bool:
        return not self.__eq__(other)
