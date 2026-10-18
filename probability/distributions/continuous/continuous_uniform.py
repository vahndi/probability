from math import sqrt
from typing import Optional

from scipy.stats import uniform

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.elicitation import elicit_continuous_uniform
from probability.distributions.mixins.attributes import AFloatDMixin, \
    BFloatDMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class ContinuousUniform(
    RVContinuous1dMixin,
    AFloatDMixin,
    BFloatDMixin,
    object
):

    """
    The continuous uniform distribution or rectangular distribution is a
    family of symmetric probability distributions. The distribution describes
    an experiment where there is an arbitrary outcome that lies between certain
    bounds.The bounds are defined by the parameters, a and b, which are the
    minimum and maximum values. The interval can be either be closed
    (eg. [a, b]) or open (eg. (a, b)).Therefore, the distribution is often
    abbreviated U (a, b), where U stands for uniform distribution.
    The difference between the bounds defines the interval length; all intervals
    of the same length on the distribution's support are equally probable. It is
    the maximum entropy probability distribution for a random variable X under
    no constraint other than that it is contained in the distribution's support.

    https://en.wikipedia.org/wiki/Continuous_uniform_distribution
    """
    def __init__(self, a: float, b: float):
        """
        Create a new continuous uniform distribution.

        :param a: Lower bound for the distribution.
        :param b: Upper bound for the distribution.
        """
        self._a = a
        self._b = b
        self._reset_distribution()

    @property
    def lower_bound(self) -> float:
        return self._a

    @property
    def upper_bound(self) -> float:
        return self._b

    def _reset_distribution(self):

        self._distribution = uniform(loc=self._a, scale=self._b - self._a)

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'ContinuousUniform':
        """
        Elicit a ContinuousUniform distribution whose central credible interval matches
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
    ) -> 'ContinuousUniform':
        """
        Elicit a ContinuousUniform distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value.
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (v2 > v1).
        """
        a, b = elicit_continuous_uniform(q1=q1, v1=v1, q2=q2, v2=v2)
        return cls(a=a, b=b)

    @classmethod
    def from_mean_and_std(cls, mean: float, std: float) -> 'ContinuousUniform':
        """
        Construct a ContinuousUniform distribution matching a given mean and standard deviation.

        :param mean: Arithmetic mean (a + b) / 2.
        :param std: Standard deviation (b - a) / sqrt(12), (std > 0).
        """
        if std <= 0.0:
            raise ValueError(f"std must be strictly positive, got {std}.")
        half_width = sqrt(3.0) * std
        return cls(a=mean - half_width, b=mean + half_width)

    @staticmethod
    def fit(data: FloatIterable,
            a: Optional[float] = None,
            b: Optional[float] = None) -> 'ContinuousUniform':
        """
        Fit a ContinuousUniform distribution to the data.

        :param data: Iterable of data to fit to.
        :param a: Optional fixed value for lower bound a.
        :param b: Optional fixed value for upper bound b.
        """
        if a is not None and b is not None:
            return ContinuousUniform(a=a, b=b)
        kwargs = {}
        if a is not None:
            kwargs['floc'] = a
        loc, scale = uniform.fit(data=data, **kwargs)
        if b is not None:
            return ContinuousUniform(a=loc, b=b)
        return ContinuousUniform(a=loc, b=loc + scale)

    def __str__(self):

        return (
            f'ContinuousUniform('
            f'a={num_format(self._a, 3)}, '
            f'b={num_format(self._b, 3)})'
        )

    def __repr__(self):

        return f'ContinuousUniform(a={self._a}, b={self._b})'

    def __eq__(self, other: 'ContinuousUniform') -> bool:

        return (
            abs(self._a - other._a) < 1e-10 and
            abs(self._b - other._b) < 1e-10
        )

    def __ne__(self, other: 'ContinuousUniform') -> bool:

        return not self.__eq__(other)
