from scipy.stats import rv_discrete, nbinom

from probability.distributions.mixins.attributes import PFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class NegativeBinomial(
    RVDiscrete1dMixin,
    PFloatDMixin,
    CalculableMixin,
    object
):
    """
    The negative binomial distribution is a discrete probability distribution
    that models the number of failures k in a sequence of independent and
    identically distributed Bernoulli trials before a specified (non-random)
    number of successes (denoted r) occurs.
    For example, we can define rolling a 6 on a die as a success, and rolling
    any other number as a failure, and ask how many failed rolls will occur
    before we see the third success (r = 3). In such a case, the probability
    distribution of the number of non-6s that appear will be a negative binomial
    distribution.

    https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """
    def __init__(self, r: int, p: float):
        """
        Create a new NegativeBinomial distribution.

        :param r: Number of successes we want.
        :param p: Probability of a failure.
        """
        self._r: int = r
        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):
        """
        https://stackoverflow.com/questions/40846992/
        alternative-parametrization-of-the-negative-binomial-in-scipy
        #comment109394209_47406400
        """
        self._distribution: rv_discrete = nbinom(self._r, 1 - self._p)

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, value: float):
        self._r = value
        self._reset_distribution()

    def __str__(self):

        return f'NegativeBinomial(r={self._r}, p={self._p})'

    def __repr__(self):

        return f'NegativeBinomial(r={self._r}, p={self._p})'
