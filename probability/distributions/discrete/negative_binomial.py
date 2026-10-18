from typing import Union

from scipy.stats import rv_discrete, nbinom

from probability.distributions.mixins.attributes import PFloatDMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin
from probability.utils import num_format, is_scalar


class NegativeBinomial(
    RVDiscrete1dMixin,
    PFloatDMixin,
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
    def __init__(self, r: float, p: float):
        """
        Create a new NegativeBinomial distribution.

        :param r: Number of successes we want.
        :param p: Probability of a failure.
        """
        self._r: float = r
        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):
        """
        https://stackoverflow.com/questions/40846992/
        alternative-parametrization-of-the-negative-binomial-in-scipy
        #comment109394209_47406400
        """
        self._distribution: rv_discrete = nbinom(
            self._r, 1 - self._p
        )

    @property
    def lower_bound(self) -> int:
        return 0

    @property
    def upper_bound(self) -> int:
        return int(self.ppf().at(0.99))

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, value: float):
        self._r = value
        self._reset_distribution()

    def mode(self) -> int:

        if self._r <= 1:
            return 0
        else:
            return int(self._p * (self._r - 1) / (1 - self._p))

    @classmethod
    def from_mean_and_variance(cls, mean: float, variance: float) -> 'NegativeBinomial':
        """
        Construct a NegativeBinomial distribution matching a given mean and variance.
        Note: Negative binomial requires overdispersion (variance > mean).

        :param mean: Mean of distribution (mean > 0).
        :param variance: Variance of distribution (variance > mean).
        """
        if mean <= 0:
            raise ValueError(f"mean must be strictly positive, got {mean}")
        if variance <= mean:
            raise ValueError(f"variance ({variance}) must be strictly greater than mean ({mean})")

        p = 1.0 - (mean / variance)
        r = (mean ** 2) / (variance - mean)
        return cls(r=r, p=p)

    @classmethod
    def fit(cls, data) -> 'NegativeBinomial':
        """
        Fit a NegativeBinomial distribution using method of moments.

        :param data: Iterable of non-negative integer counts.
        """
        import numpy as np
        arr = np.asarray(list(data), dtype=float)
        mean = float(np.mean(arr))
        variance = float(np.var(arr, ddof=1)) if len(arr) > 1 else mean * 1.5
        if variance <= mean:
            # minimal overdispersion fallback
            variance = mean * 1.01
        return cls.from_mean_and_variance(mean=mean, variance=variance)

    def __str__(self):

        return (
            f'NegativeBinomial('
            f'r={num_format(self._r, 3)}, '
            f'p={num_format(self._p, 3)})'
        )

    def __repr__(self):

        return f'NegativeBinomial(r={self._r}, p={self._p})'

    def __eq__(self, other: Union['NegativeBinomial', int, float]):

        if is_scalar(other):
            return self.pmf().at(other)
        else:
            return (
                self._r == other._r and
                abs(self._p - other._p) < 1e-10
            )

    def __ne__(
            self, other: Union['NegativeBinomial', int, float]
    ) -> Union[bool, float]:

        if is_scalar(other):
            return 1 - self.pmf().at(other)
        else:
            return not self.__eq__(other)
