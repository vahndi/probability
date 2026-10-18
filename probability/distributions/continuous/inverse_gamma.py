from typing import Optional

from scipy.stats import invgamma, rv_continuous

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.elicitation import elicit_inverse_gamma
from probability.distributions.mixins.attributes import AlphaFloatDMixin, \
    BetaFloatDMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class InverseGamma(
    RVContinuous1dMixin,
    AlphaFloatDMixin,
    BetaFloatDMixin,
    object
):

    """
    The inverse gamma distribution is a two-parameter family of continuous
    probability distributions on the positive real line, which is the
    distribution of the reciprocal of a variable distributed according to the
    gamma distribution.

    Perhaps the chief use of the inverse gamma distribution is in Bayesian
    statistics, where the distribution arises as the marginal posterior
    distribution for the unknown variance of a normal distribution, if an
    uninformative prior is used, and as an analytically tractable conjugate
    prior, if an informative prior is required.

    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    def __init__(self, alpha: float, beta: float):

        self._alpha: float = alpha
        self._beta: float = beta
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = invgamma(
            a=self._alpha, scale=self._beta
        )

    def mode(self) -> float:
        return self.beta / (self.alpha + 1)

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.isf().at(0.01)

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'InverseGamma':
        """
        Elicit an InverseGamma distribution whose central credible interval matches
        [lower, upper] with probability `prob`.

        :param lower: Lower bound of interval (lower > 0).
        :param upper: Upper bound of interval (upper > lower).
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
    ) -> 'InverseGamma':
        """
        Elicit an InverseGamma distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value (v1 > 0).
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (v2 > v1).
        """
        alpha, beta = elicit_inverse_gamma(q1=q1, v1=v1, q2=q2, v2=v2)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def from_mean_and_std(cls, mean: float, std: float) -> 'InverseGamma':
        """
        Construct an InverseGamma distribution matching a given mean and standard deviation.
        Note: Requires alpha > 2 for finite variance, which holds for any mean > 0, std > 0.

        :param mean: Arithmetic mean E[X] (mean > 0).
        :param std: Standard deviation SD[X] (std > 0).
        """
        if mean <= 0.0:
            raise ValueError(f"mean must be strictly positive, got {mean}.")
        if std <= 0.0:
            raise ValueError(f"std must be strictly positive, got {std}.")

        var = std ** 2
        alpha = 2.0 + (mean ** 2) / var
        beta = mean * (alpha - 1.0)
        return cls(alpha=alpha, beta=beta)

    @staticmethod
    def fit(data: FloatIterable,
            alpha: Optional[float] = None,
            beta: Optional[float] = None) -> 'InverseGamma':
        """
        Fit an Inverse Gamma distribution to the data.

        :param data: Iterable of data to fit to.
        :param alpha: Optional fixed value for alpha.
        :param beta: Optional fixed value for beta.
        """
        kwargs = {}
        for arg, kw in zip(
            (alpha, beta),
            ('fa', 'fscale')
        ):
            if arg is not None:
                kwargs[kw] = arg
        alpha, loc, scale = invgamma.fit(data=data, floc=0, **kwargs)
        return InverseGamma(alpha=alpha, beta=scale)

    def __str__(self):

        return (
            f'InverseGamma('
            f'α={num_format(self._alpha, 3)}, '
            f'β={num_format(self._beta, 3)})'
        )

    def __repr__(self):

        return f'InverseGamma(alpha={self._alpha}, beta={self._beta})'

    def __eq__(self, other: 'InverseGamma') -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )

    def __ne__(self, other: 'InverseGamma') -> bool:

        return not self.__eq__(other)
