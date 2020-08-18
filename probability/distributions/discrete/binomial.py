from scipy.stats import binom, rv_discrete

from probability.distributions.mixins.attributes import NIntDMixin, PFloatDMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class Binomial(
    RVDiscrete1dMixin,
    NIntDMixin,
    PFloatDMixin,
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

    def __str__(self):

        return f'Binomial(n={self._n}, p={self._p})'

    def __repr__(self):

        return f'Binomial(n={self._n}, p={self._p})'

    def __eq__(self, other: 'Binomial'):

        return (
            self._n == other._n and
            abs(self._p - other._p) < 1e-10
        )
