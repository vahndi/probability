from typing import Union

from scipy.stats import poisson, rv_discrete

from probability.distributions.mixins.attributes import LambdaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin
from probability.utils import num_format


class Poisson(
    RVDiscrete1dMixin,
    LambdaFloatDMixin,
    CalculableMixin,
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
        Create a new Poisson distribution.

        :param lambda_: Average rate at which events occur.
        """
        self._lambda: float = lambda_
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = poisson(self._lambda)

    @property
    def lower_bound(self) -> int:
        return 1

    @property
    def upper_bound(self) -> int:
        return int(self.isf().at(0.01))

    def mode(self) -> int:

        return int(self._lambda)

    def __str__(self):

        return f'Poisson(λ={num_format(self._lambda, 3)})'

    def __repr__(self):

        return f'Poisson(lambda_={self._lambda})'

    def __eq__(self, other: Union['Poisson', int, float]):

        if type(other) in (int, float):
            return self.pmf().at(other)
        else:
            return abs(self._lambda - other._lambda) < 1e-10

    def __ne__(
            self, other: Union['Poisson', int, float]
    ) -> Union[bool, float]:

        if type(other) in (int, float):
            return 1 - self.pmf().at(other)
        else:
            return not self.__eq__(other)
