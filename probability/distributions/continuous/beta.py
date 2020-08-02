from scipy.stats import beta as beta_dist, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.special import prob_bb_greater_exact


class Beta(RVContinuous1dMixin):
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

        return f'Beta(α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'

    def __gt__(self, other: 'Beta') -> float:

        return prob_bb_greater_exact(
            alpha_1=self._alpha, beta_1=self._beta, m_1=0, n_1=0,
            alpha_2=other._alpha, beta_2=other._beta, m_2=0, n_2=0
        )

    def __lt__(self, other: 'Beta') -> float:

        return other < self

    def __eq__(self, other: 'Beta') -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )
