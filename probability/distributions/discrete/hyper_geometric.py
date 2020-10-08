from scipy.stats import rv_discrete, hypergeom

from probability.distributions.mixins.attributes import NIntDMixin, \
    BigNIntDMixin, BigKIntDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin


class HyperGeometric(
    RVDiscrete1dMixin,
    BigNIntDMixin,
    BigKIntDMixin,
    NIntDMixin,
    CalculableMixin,
    object
):
    """
    The hyper-geometric distribution is a discrete probability distribution that
    describes the probability of k successes (random draws for which the object
    drawn has a specified feature) in n draws, without replacement, from a
    finite population of size N that contains exactly K objects with that
    feature, wherein each draw is either a success or a failure.
    In contrast, the binomial distribution describes the probability of k
    successes in n draws with replacement.

    https://en.wikipedia.org/wiki/Hypergeometric_distribution
    """
    def __init__(self, N: int, K: int, n: int):
        """
        Create a new hyper-geometric distribution.

        :param N: Population size.
        :param K: Number of objects with a given feature.
        :param n: Number of trials.
        """
        self._N: int = N
        self._K: int = K
        self._n: int = n
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = hypergeom(
            self._N, self._K, self._n
        )

    def __str__(self):

        return f'HyperGeometric(N={self._N}, K={self._K}, n={self._n})'

    def __repr__(self):

        return f'HyperGeometric(N={self._N}, K={self._K}, n={self._n})'

    def __eq__(self, other: 'HyperGeometric'):

        return (
            self._N == other._N and
            self._K == other._K and
            self._n == other._n
        )
