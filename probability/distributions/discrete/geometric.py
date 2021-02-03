from typing import Union

from scipy.stats import geom, rv_discrete

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import PFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin
from probability.utils import num_format


class Geometric(
    RVDiscrete1dMixin,
    PFloatDMixin,
    CalculableMixin,
    object
):
    """
    The (shifted) geometric distribution gives the probability that the first
    occurrence of success requires k independent trials, each with success
    probability p.

    https://en.wikipedia.org/wiki/Geometric_distribution
    """
    def __init__(self, p: float):
        """
        Create a new shifted geometric distribution.

        :param p: Probability of success in any individual trial.
        """
        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = geom(self._p)

    @property
    def lower_bound(self) -> int:
        return 1

    @property
    def upper_bound(self) -> int:
        return int(self.ppf().at(0.99))

    def mode(self) -> int:

        return 1

    @staticmethod
    def fit(data: FloatIterable) -> 'Geometric':
        """
        Fit a Geometric distribution to the data of one or more experiments
        using the maximum likelihood estimate for p.

        :param data: Iterable of data to fit to. Each result represents the
                     result of a single trial, and should be 0 or 1.
        """
        n = sum(data)  # number of experiments
        k = len(data)  # number of trials
        p = n / k
        return Geometric(p=p)

    @staticmethod
    def fits(data: FloatIterable) -> 'Geometric':
        """
        Fit a Geometric distribution to the distribution of results of a
        series of experiments using the method of moments.

        https://en.wikipedia.org/wiki/Geometric_distribution#Parameter_estimation

        :param data: Iterable of results to fit to. Each value represents the
                     number of trials in an experiment to get 1 success, and
                     should be at least 1.
        """
        n = len(data)  # number of experiments
        k = sum(data)  # number of trials
        p = n / k
        return Geometric(p=p)

    def __str__(self):

        return f'Geometric(p={num_format(self._p, 3)})'

    def __repr__(self):

        return f'Geometric(p={self._p})'

    def __eq__(self, other: Union['Geometric', int, float]):

        if type(other) in (int, float):
            return self.pmf().at(other)
        else:
            return abs(self._p - other._p) < 1e-10

    def __ne__(
            self, other: Union['Geometric', int, float]
    ) -> Union[bool, float]:

        if type(other) in (int, float):
            return 1 - self.pmf().at(other)
        else:
            return not self.__eq__(other)
