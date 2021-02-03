from typing import Union

from scipy.stats import binom, rv_discrete

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import NIntDMixin, PFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin
from probability.utils import num_format


class Binomial(
    RVDiscrete1dMixin,
    NIntDMixin,
    PFloatDMixin,
    CalculableMixin,
    object
):
    """
    The binomial distribution with parameters n and p is the discrete
    probability distribution of the number of successes in a sequence of n
    independent experiments, each asking a yes–no question, and each with its
    own boolean-valued outcome: success/yes/true/one (with probability p) or
    failure/no/false/zero (with probability q = 1 − p)

    https://en.wikipedia.org/wiki/Binomial_distribution
    """
    def __init__(self, n: int, p: float):
        """
        Create a new binomial distribution.

        :param n: Number of trials.
        :param p: Probability of success in an individual trial.
        """
        self._n: int = n
        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = binom(self._n, self._p)

    def mode(self) -> int:

        return int(self._n * self._p)

    @property
    def lower_bound(self) -> int:
        return 0

    @property
    def upper_bound(self) -> int:
        return self._n

    @staticmethod
    def fit(data: FloatIterable) -> 'Binomial':
        """
        Fit a Binomial distribution to the data from a single experiment using
        the maximum likelihood estimate for p.

        https://en.wikipedia.org/wiki/Binomial_distribution#Estimation_of_parameters

        :param data: Iterable of data of to fit to. Each value should be 0 or 1.
        """
        n = len(data)
        k = sum(data)
        return Binomial(n=n, p=k / n)

    @staticmethod
    def fits(data: FloatIterable, n: int) -> 'Binomial':
        """
        Fit a Binomial distribution to the distribution of results of a series
        of experiments using the maximum likelihood estimate for p.

        :param data: Iterable of data to fit to.
        :param n: Number of trials.
        """
        N = n * len(data)
        k = sum(data)
        return Binomial(n=n, p=k / N)

    def __str__(self):

        return (
            f'Binomial('
            f'n={num_format(self._n, 3)}, '
            f'p={num_format(self._p, 3)})'
        )

    def __repr__(self):

        return f'Binomial(n={self._n}, p={self._p})'

    def __eq__(
            self, other: Union['Binomial', int, float]
    ) -> Union[bool, float]:

        if type(other) in (int, float):
            return self.pmf().at(other)
        else:
            return (
                self._n == other._n and
                abs(self._p - other._p) < 1e-10
            )

    def __ne__(
            self, other: Union['Binomial', int, float]
    ) -> Union[bool, float]:

        if type(other) in (int, float):
            return 1 - self.pmf().at(other)
        else:
            return not self.__eq__(other)
