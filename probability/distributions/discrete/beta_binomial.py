from typing import Union

from numpy import power, sum
from scipy.stats import betabinom, rv_discrete

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import NIntDMixin, \
    AlphaFloatDMixin, BetaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin
from probability.utils import num_format


class BetaBinomial(
    RVDiscrete1dMixin,
    NIntDMixin,
    AlphaFloatDMixin,
    BetaFloatDMixin,
    CalculableMixin,
    object
):
    """
    The beta-binomial distribution is a family of discrete probability
    distributions on a finite support of non-negative integers arising when the
    probability of success in each of a fixed or known number of Bernoulli
    trials is either unknown or random.
    The beta-binomial distribution is the binomial distribution in which the
    probability of success at each of n trials is not fixed but randomly drawn
    from a beta distribution.
    It reduces to the Bernoulli distribution as a special case when n = 1.
    For α = β = 1, it is the discrete uniform distribution from 0 to n.
    It also approximates the binomial distribution arbitrarily well for large α
    and β.
    Similarly, it contains the negative binomial distribution in the limit with
    large β and n.
    The beta-binomial is a one-dimensional version of the Dirichlet-multinomial
    distribution as the binomial and beta distributions are univariate versions
    of the multinomial and Dirichlet distributions respectively.

    https://en.wikipedia.org/wiki/Beta-binomial_distribution
    """

    def __init__(self, n: int, alpha: float, beta: float):
        """
        Create a new beta-binomial distribution.

        :param n: Number of trials.
        :param alpha: α parameter for the probability of the binomial.
        :param beta: β parameter for the probability of the binomial.
        """
        self._n: int = n
        self._alpha = alpha
        self._beta = beta
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_discrete = betabinom(
            n=self._n, a=self._alpha, b=self._beta
        )

    @property
    def lower_bound(self) -> int:
        return 0

    @property
    def upper_bound(self) -> int:
        return self._n

    @staticmethod
    def fit(data: FloatIterable):
        raise NotImplementedError

    @staticmethod
    def fits(data: FloatIterable, n: int) -> 'BetaBinomial':
        """
        Fit a BetaBinomial distribution to the distribution of results of a
        series of N experiments, each having n trials, using the method of
        moments.

        https://en.wikipedia.org/wiki/Beta-binomial_distribution#Method_of_moments

        :param data: Number of successes in each trial.
        :param n: Number of trials per experiment.
        """
        N = len(data)
        m1 = sum(data) / N
        m2 = sum(power(data, 2)) / N
        denominator = n * ((m2 / m1) - m1 - 1) + m1
        alpha = (n * m1 - m2) / denominator
        beta = (n - m1) * (n - m2 / m1) / denominator
        return BetaBinomial(n=n, alpha=alpha, beta=beta)

    def __str__(self):
        return f'BetaBinomial(' \
               f'n={self._n}, ' \
               f'α={num_format(self._alpha, 3)}, ' \
               f'β={num_format(self._beta, 3)})'

    def __repr__(self):
        return f'BetaBinomial(' \
               f'n={self._n}, ' \
               f'alpha={self._alpha}, ' \
               f'beta={self._beta})'

    def __eq__(self, other: Union['BetaBinomial', int, float]):

        if type(other) in (int, float):
            return self.pmf().at(other)
        else:
            return (
                self._n == other._n and
                abs(self._alpha - other._alpha) < 1e-10 and
                abs(self._beta - other._beta) < 1e-10
            )

    def __ne__(
            self, other: Union['BetaBinomial', int, float]
    ) -> Union[bool, float]:

        if type(other) in (int, float):
            return 1 - self.pmf().at(other)
        else:
            return not self.__eq__(other)
