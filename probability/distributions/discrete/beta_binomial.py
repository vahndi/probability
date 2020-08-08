from scipy.stats import betabinom, rv_discrete

from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin


class BetaBinomial(RVDiscrete1dMixin):
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
            self._n, self._alpha, self._beta
        )

    @property
    def n(self) -> float:
        return self._n

    @n.setter
    def n(self, value: float):
        self._n = value
        self._reset_distribution()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self._reset_distribution()

    def __str__(self):
        return f'BetaBinomial(' \
               f'n={self._n}, ' \
               f'α={self._alpha}, ' \
               f'β={self._beta})'

    def __repr__(self):
        return f'BetaBinomial(' \
               f'n={self._n}, ' \
               f'alpha={self._alpha}, ' \
               f'beta={self._beta})'

    def __eq__(self, other: 'BetaBinomial'):
        return (
            self._n == other._n and
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )
