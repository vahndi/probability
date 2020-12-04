from typing import Union

from scipy.stats import beta as beta_dist, rv_continuous

from probability.distributions.mixins.attributes import AlphaFloatDMixin, \
    BetaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.domains import FixedDomainMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from probability.distributions.special import prob_bb_greater_exact
from probability.utils import num_format


class Beta(
    RVContinuous1dMixin,
    AlphaFloatDMixin,
    BetaFloatDMixin,
    CalculableMixin,
    FixedDomainMixin,
    object
):
    """
    The beta distribution is a family of continuous probability distributions
    defined on the interval [0, 1] parameterized by two positive shape
    parameters, denoted by α and β, that appear as exponents of the random
    variable and control the shape of the distribution.
    The generalization to multiple variables is called a Dirichlet distribution.

    The beta distribution is a suitable model for the random behavior of
    percentages and proportions.

    https://en.wikipedia.org/wiki/Beta_distribution
    """
    lower_bound = 0
    upper_bound = 1

    def __init__(self, alpha: float, beta: float):
        """
        Create a new beta distribution.

        :param alpha: First shape parameter: Interpretation can be # successes.
        :param beta: Second shape parameter: : Interpretation can be # failures.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = beta_dist(self._alpha, self._beta)

    def __str__(self):

        return f'Beta(' \
               f'α={num_format(self._alpha, 3)}, ' \
               f'β={num_format(self._beta, 3)})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'

    def __gt__(self, other: Union['Beta', float]) -> float:

        if isinstance(other, float):
            return (self.rvs(NUM_SAMPLES_COMPARISON) > other).mean()
        elif isinstance(other, Beta):
            return prob_bb_greater_exact(
                alpha_1=self._alpha, beta_1=self._beta, m_1=0, n_1=0,
                alpha_2=other._alpha, beta_2=other._beta, m_2=0, n_2=0
            )
        else:
            raise TypeError('other must be of type float or Rvs1dMixin')

    def __lt__(self, other: Union['Beta', float]) -> float:

        return other > self

    def __eq__(self, other: Union['Beta', float]) -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )
