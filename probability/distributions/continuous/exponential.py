from scipy.stats import expon, rv_continuous

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import LambdaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


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

        self._distribution: rv_continuous = expon(
            loc=0, scale=1 / self._lambda
        )

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    def mode(self) -> float:
        return 0.0

    @staticmethod
    def fit(data: FloatIterable) -> 'Exponential':
        """
        Fit an Exponential distribution to the data.

        :param data: Iterable of data to fit to.
        """
        loc, scale = expon.fit(data=data, floc=0)
        return Exponential(lambda_=1 / scale)

    def __str__(self):

        return f'Exponential(λ={num_format(self._lambda, 3)})'

    def __repr__(self):

        return f'Exponential(lambda_={self._lambda})'

    def __eq__(self, other: 'Exponential') -> bool:

        return abs(self._lambda - other._lambda) < 1e-10

    def __ne__(self, other: 'Exponential') -> bool:

        return not self.__eq__(other)
