from scipy.stats import uniform

from probability.distributions.mixins.attributes import AFloatDMixin, \
    BFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin


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

    def _reset_distribution(self):

        self._distribution = uniform(loc=self._a, scale=self._b - self._a)

    def __str__(self):

        return f'ContinuousUniform(a={self._a: 0.2f}, b={self._b: 0.2f})'

    def __repr__(self):

        return f'ContinuousUniform(a={self._a}, b={self._b})'
