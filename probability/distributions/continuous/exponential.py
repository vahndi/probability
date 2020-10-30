from scipy.stats import expon, rv_continuous

from probability.distributions.mixins.attributes import LambdaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin


class Exponential(
    RVContinuous1dMixin,
    LambdaFloatDMixin,
    CalculableMixin,
    object
):
    """
    The exponential distribution is the probability distribution of the time
    between events in a Poisson point process, i.e., a process in which events
    occur continuously and independently at a constant average rate.
    It is a particular case of the gamma distribution.
    It is the continuous analogue of the geometric distribution,
    and it has the key property of being memory-less.

    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    def __init__(self, lambda_: float):
        """
        Create a new exponential distribution.

        :param lambda_: The average rate at which events occur.
        """
        self._lambda: float = lambda_
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = expon(loc=0, scale=1 / self._lambda)

    def __str__(self):

        return f'Exponential(λ={self._lambda})'

    def __repr__(self):

        return f'Exponential(lambda_={self._lambda})'
