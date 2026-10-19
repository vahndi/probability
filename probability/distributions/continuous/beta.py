from typing import Union, Tuple, Optional

from scipy.stats import beta as beta_dist, rv_continuous

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.elicitation import elicit_beta
from probability.distributions.mixins.attributes import \
    AlphaFloatDMixin, BetaFloatDMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.special import prob_bb_greater_exact
from probability.utils import num_format


class Beta(
    RVContinuous1dMixin,
    AlphaFloatDMixin,
    BetaFloatDMixin,
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
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return 1.0

    def mode(self) -> Optional[Union[float, Tuple[float, float]]]:

        alpha = self.alpha
        beta = self.beta
        if alpha == 1 and beta == 1:
            return None
        elif alpha < 1 and beta < 1:
            return 0, 1
        elif alpha > 1 and beta > 1:
            return (alpha - 1) / (alpha + beta - 2)
        elif alpha <= 1 < beta:
            return 0
        elif alpha > 1 >= beta:
            return 1
        else:
            raise ValueError("Can't calculate mode for this distribution.")

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'Beta':
        """
        Elicit a Beta distribution whose central credible interval matches
        [lower, upper] with probability `prob`.

        :param lower: Lower bound of interval (0 < lower < 1).
        :param upper: Upper bound of interval (0 < lower < upper < 1).
        :param prob: Central coverage probability (0 < prob < 1), default 0.95.
        """
        if not (0.0 < prob < 1.0):
            raise ValueError(f"prob must be between 0 and 1, got {prob}.")
        alpha_tail = (1.0 - prob) / 2.0
        return cls.from_quantiles(
            q1=alpha_tail, v1=lower,
            q2=1.0 - alpha_tail, v2=upper
        )

    @classmethod
    def from_quantiles(
            cls, q1: float, v1: float, q2: float, v2: float
    ) -> 'Beta':
        """
        Elicit a Beta distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value (0 < v1 < 1).
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (0 < v1 < v2 < 1).
        """
        alpha, beta = elicit_beta(q1=q1, v1=v1, q2=q2, v2=v2)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def from_mean_and_std(cls, mean: float, std: float) -> 'Beta':
        """
        Construct a Beta distribution matching a given mean and standard deviation.

        :param mean: Arithmetic mean (0 < mean < 1).
        :param std: Standard deviation (0 < std < sqrt(mean * (1 - mean))).
        """
        if not (0.0 < mean < 1.0):
            raise ValueError(f"mean must be between 0 and 1, got {mean}.")
        if std <= 0.0:
            raise ValueError(f"std must be strictly positive, got {std}.")

        var = std ** 2
        max_var = mean * (1.0 - mean)
        if var >= max_var:
            raise ValueError(
                f"Variance ({var}) must be strictly less than mean*(1-mean) ({max_var}) for a Beta distribution."
            )

        nu = (max_var / var) - 1.0
        alpha = mean * nu
        beta = (1.0 - mean) * nu
        return cls(alpha=alpha, beta=beta)

    @staticmethod
    def fit(data: FloatIterable,
            alpha: Optional[float] = None,
            beta: Optional[float] = None) -> 'Beta':
        """
        Fit a Beta distribution to the data.

        :param data: Iterable of data to fit to.
        :param alpha: Optional fixed value for alpha.
        :param beta: Optional fixed value for beta.
        """
        kwargs = {}
        for arg, kw in zip(
            (alpha, beta),
            ('fa', 'fb')
        ):
            if arg is not None:
                kwargs[kw] = arg
        alpha, beta, loc, scale = beta_dist.fit(
            data=data,
            floc=-1e-9, fscale=1 + 2e-9,  # avoid fit error when 0 or 1 in data
            **kwargs
        )
        return Beta(alpha=alpha, beta=beta)

    def __str__(self):

        return f'Beta(' \
               f'α={num_format(self._alpha, 3)}, ' \
               f'β={num_format(self._beta, 3)})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'

    def __gt__(self, other: Union['Beta', float, int]) -> float:

        if isinstance(other, Beta):
            return prob_bb_greater_exact(
                alpha_1=self._alpha, beta_1=self._beta, m_1=0, n_1=0,
                alpha_2=other._alpha, beta_2=other._beta, m_2=0, n_2=0
            )
        else:
            return super(Beta, self).__gt__(other)

    def __ge__(self, other: Union['Beta', float, int]) -> float:

        if isinstance(other, Beta):
            return self.__gt__(other)
        else:
            return super(Beta, self).__ge__(other)

    def __lt__(self, other: Union['Beta', float, int]) -> float:

        if isinstance(other, Beta):
            return other > self
        else:
            return super(Beta, self).__lt__(other)

    def __le__(self, other: Union['Beta', float, int]) -> float:

        if isinstance(other, Beta):
            return other > self
        else:
            return super(Beta, self).__le__(other)

    def __eq__(self, other: 'Beta') -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )

    def __ne__(self, other: 'Beta') -> bool:

        return not self.__eq__(other)
