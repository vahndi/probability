from typing import Optional

from scipy.stats import rv_continuous, gamma

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.elicitation import elicit_gamma
from probability.distributions.mixins.attributes import AlphaFloatDMixin, \
    BetaFloatDMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class Gamma(
    RVContinuous1dMixin,
    AlphaFloatDMixin,
    BetaFloatDMixin,
    object
):

    """
    The gamma distribution is a two-parameter family of continuous probability
    distributions. The exponential distribution, Erlang distribution, and
    chi-squared distribution are special cases of the gamma distribution.

    The parameterization with k and θ appears to be more common in econometrics
    and certain other applied fields, where for example the gamma distribution
    is frequently used to model waiting times.

    The parameterization with α and β is more common in Bayesian statistics,
    where the gamma distribution is used as a conjugate prior distribution for
    various types of inverse scale (rate) parameters, such as the λ (rate) of
    an exponential distribution or of a Poisson distribution

    https://en.wikipedia.org/wiki/Gamma_distribution
    """

    _parameterization: str

    def __init__(self, alpha: float, beta: float,
                 parameterization: str = 'αβ'):
        """
        Create a new gamma distribution.

        :param alpha: Shape parameter: Interpretation can be number of events
                      that we are waiting on to happen. When alpha = 1 we get
                      the exponential distribution.
        :param beta: Rate parameter: Interpretation can be the rate at which
                     events happen.
        :param parameterization: One of ['αβ', 'kθ']
        """
        assert parameterization in ('αβ', 'kθ')
        self._alpha: float = alpha
        self._beta: float = beta
        self._parameterization: str = parameterization
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = gamma(
            a=self._alpha, scale=1 / self._beta
        )

    @staticmethod
    def from_alpha_beta(alpha: float, beta: float) -> 'Gamma':

        return Gamma(alpha=alpha, beta=beta,
                     parameterization='αβ')

    @staticmethod
    def from_k_theta(k: float, theta: float) -> 'Gamma':

        return Gamma(alpha=k, beta=1 / theta,
                     parameterization='kθ')

    @property
    def k(self) -> float:
        return self._alpha

    @k.setter
    def k(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def theta(self):
        return 1 / self._beta

    @theta.setter
    def theta(self, value: float):
        self._beta = 1 / value
        self._reset_distribution()

    def mode(self) -> float:
        if self.alpha >= 1:
            return (self.alpha - 1) / self.beta
        else:
            raise ValueError("Can't calculate mode for this distribution.")

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    @classmethod
    def from_interval(
            cls, lower: float, upper: float, prob: float = 0.95
    ) -> 'Gamma':
        """
        Elicit a Gamma distribution whose central credible interval matches
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
    ) -> 'Gamma':
        """
        Elicit a Gamma distribution matching two arbitrary cumulative quantiles:
            P(X <= v1) = q1 and P(X <= v2) = q2

        :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
        :param v1: Lower quantile value (v1 > 0).
        :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
        :param v2: Upper quantile value (v2 > v1).
        """
        alpha, beta = elicit_gamma(q1=q1, v1=v1, q2=q2, v2=v2)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def from_mean_and_std(cls, mean: float, std: float) -> 'Gamma':
        """
        Construct a Gamma distribution matching a given mean and standard deviation.

        :param mean: Arithmetic mean E[X] (mean > 0).
        :param std: Standard deviation SD[X] (std > 0).
        """
        if mean <= 0.0:
            raise ValueError(f"mean must be strictly positive, got {mean}.")
        if std <= 0.0:
            raise ValueError(f"std must be strictly positive, got {std}.")

        var = std ** 2
        beta = mean / var
        alpha = mean * beta
        return cls(alpha=alpha, beta=beta)

    @staticmethod
    def fit(data: FloatIterable,
            alpha: Optional[float] = None,
            beta: Optional[float] = None) -> 'Gamma':
        """
        Fit a Gamma distribution to the data.

        :param data: Iterable of data to fit to.
        :param alpha: Optional fixed value for alpha.
        :param beta: Optional fixed value for beta.
        """
        if alpha is not None and beta is not None:
            return Gamma(alpha=alpha, beta=beta)
        kwargs = {}
        if alpha is not None:
            kwargs['f0'] = alpha
        if beta is not None:
            kwargs['fscale'] = 1 / beta
        alpha, loc, scale = gamma.fit(data=data, floc=0, **kwargs)
        return Gamma(alpha=alpha, beta=1 / scale)

    def __str__(self):

        if self._parameterization == 'αβ':
            return (
                f'Gamma('
                f'α={num_format(self._alpha, 3)}, '
                f'β={num_format(self._beta, 3)})'
            )
        else:
            return (
                f'Gamma('
                f'k={num_format(self._alpha, 3)}, '
                f'θ={num_format(1 / self._beta, 3)})'
            )

    def __repr__(self):

        if self._parameterization == 'αβ':
            return f'Gamma(alpha={self._alpha}, beta={self._beta})'
        else:
            return f'Gamma(k={self._alpha}, theta={1 / self._beta})'

    def __eq__(self, other: 'Gamma') -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )

    def __ne__(self, other: 'Gamma') -> bool:

        return not self.__eq__(other)
