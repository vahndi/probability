from typing import Optional

from scipy.stats import uniform

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import AFloatDMixin, \
    BFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class ContinuousUniform(
    RVContinuous1dMixin,
    AFloatDMixin,
    BFloatDMixin,
    CalculableMixin,
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
        kwargs = {}
        for arg, kw in zip(
            (a, b),
            ('fa', 'fb')
        ):
            if arg is not None:
                kwargs[kw] = arg
        loc, scale = uniform.fit(data=data, **kwargs)
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
