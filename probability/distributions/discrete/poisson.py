from scipy.stats import poisson, rv_discrete

from probability.distributions.mixins.attributes import LambdaFloatDMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class Poisson(
    RVDiscrete1dMixin,
    LambdaFloatDMixin,
    object
):
    """
    The Poisson distribution is a discrete probability distribution that
    expresses the probability of a given number of events occurring in a fixed
    interval of time or space if these events occur with a known constant mean
    rate and independently of the time since the last event.
    The Poisson distribution can also be used for the number of events in other
    specified intervals such as distance, area or volume.

    https://en.wikipedia.org/wiki/Poisson_distribution
    """
    def __init__(self, lambda_: float):
        """
        Create a new poisson distribution.

        :param lambda_: Average rate at which events occur.
        """
        self._lambda: float = lambda_
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = poisson(self._lambda)

    def __str__(self):

        return f'Poisson(λ={self._lambda})'

    def __repr__(self):

        return f'Poisson(lambda_={self._lambda})'
