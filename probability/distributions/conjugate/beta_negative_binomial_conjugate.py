from typing import Optional, Union, List

from matplotlib.figure import Figure
from pandas import Series, DataFrame

from probability.distributions.conjugate.priors import UniformPrior
from probability.distributions.continuous.beta import Beta
from probability.distributions.discrete.negative_binomial import NegativeBinomial
from probability.distributions.mixins.attributes import (
    AlphaFloatMixin,
    BetaFloatMixin,
    NIntMixin,
    KIntMixin,
)
from probability.distributions.mixins.conjugate import ConjugateMixin
from probability.plotting import new_subplots
from probability.supports import SUPPORT_BETA
from probability.utils import num_format


class BetaNegativeBinomialConjugate(
    ConjugateMixin,
    AlphaFloatMixin,
    BetaFloatMixin,
    NIntMixin,
    KIntMixin,
    object
):
    """
    Class for calculating Bayesian conjugate probabilities for the
    Negative Binomial distribution with a Beta prior on the failure probability p.

    Model:
    - In each experiment, trials continue until r successes occur.
    - Let p in [0, 1] be the failure probability (probability of non-success in a trial).
    - The number of failures k follows NegativeBinomial(r, p):
        P(K = k | p) = C(k + r - 1, k) * (1 - p)^r * p^k
    - The conjugate prior on p is Beta(alpha, beta).
    - After observing n experiments with total k failures:
        Posterior is Beta(alpha' = alpha + k, beta' = beta + n * r).

    Links:
    - https://en.wikipedia.org/wiki/Negative_binomial_distribution
    - https://en.wikipedia.org/wiki/Conjugate_prior
    """

    def __init__(self, r: int, k: int, n: int = 1,
                 alpha: float = UniformPrior.NegativeBinomial.alpha,
                 beta: float = UniformPrior.NegativeBinomial.beta):
        """
        Create a new BetaNegativeBinomialConjugate model.

        :param r: Target number of successes per experiment (r >= 1).
        :param k: Total number of failures observed across experiments (k >= 0).
        :param n: Number of experiments observed (n >= 1), default 1.
        :param alpha: Prior Beta shape alpha (alpha > 0).
        :param beta: Prior Beta shape beta (beta > 0).
        """
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"alpha and beta must be positive, got alpha={alpha}, beta={beta}")

        self._r: int = int(r)
        self._k: int = int(k)
        self._n: int = int(n)
        self._alpha: float = float(alpha)
        self._beta: float = float(beta)

    @property
    def r(self) -> int:
        return self._r

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._k

    @property
    def beta_prime(self) -> float:
        return self._beta + (self._n * self._r)

    def prior(self) -> Beta:
        """
        Return the prior Beta distribution over failure probability p.
        """
        return Beta(
            alpha=self._alpha,
            beta=self._beta
        ).with_y_label(
            r'$P(p \mid \alpha, \beta)$'
        ).prepend_to_label('Prior: ')

    def likelihood(self) -> NegativeBinomial:
        """
        Return the maximum-likelihood NegativeBinomial distribution for the data.
        """
        total_r = self._r * self._n
        total_trials = self._k + total_r
        p_mle = self._k / total_trials if total_trials > 0 else 0.5
        return NegativeBinomial(r=total_r, p=p_mle)

    def posterior(self) -> Beta:
        """
        Return the posterior Beta distribution over failure probability p.
        """
        return Beta(
            alpha=self.alpha_prime,
            beta=self.beta_prime
        ).with_y_label(
            r'$P(p \mid \alpha^\prime, \beta^\prime)$'
        ).prepend_to_label('Posterior: ')

    def plot(self, **kwargs) -> Figure:
        """
        Plot prior, likelihood, and posterior distributions.
        """
        fig, axes = new_subplots(n_rows=1, n_cols=3)
        ax_prior, ax_lik, ax_post = axes.flat

        self.prior().plot(x=SUPPORT_BETA, ax=ax_prior, **kwargs)
        ax_prior.set_title('Prior')
        ax_prior.legend()

        k_max = max(int(self._k * 1.5) + 5, 10)
        self.likelihood().plot(k=range(k_max), ax=ax_lik, **kwargs)
        ax_lik.set_title(f'Likelihood (n={self._n}, r={self._r})')
        ax_lik.legend()

        self.posterior().plot(x=SUPPORT_BETA, ax=ax_post, **kwargs)
        ax_post.set_title('Posterior')
        ax_post.legend()

        return fig

    @staticmethod
    def infer_posterior(data: Union[Series, List[int]],
                        r: int,
                        alpha: Optional[float] = None,
                        beta: Optional[float] = None) -> Beta:
        """
        Infer the posterior Beta distribution given observed failure counts.

        :param data: List or Series of failure counts, one per experiment.
        :param r: Number of target successes per experiment.
        :param alpha: Prior alpha (default Uniform).
        :param beta: Prior beta (default Uniform).
        """
        if alpha is None:
            alpha = UniformPrior.NegativeBinomial.alpha
        if beta is None:
            beta = UniformPrior.NegativeBinomial.beta

        data_list = [int(x) for x in data]
        k_total = sum(data_list)
        n_experiments = len(data_list)

        return BetaNegativeBinomialConjugate(
            r=r, k=k_total, n=n_experiments,
            alpha=alpha, beta=beta
        ).posterior()

    def __str__(self):
        return (
            f'BetaNegativeBinomialConjugate('
            f'r={self._r}, k={self._k}, n={self._n}, '
            f'α={num_format(self._alpha, 3)}, β={num_format(self._beta, 3)})'
        )

    def __repr__(self):
        return (
            f'BetaNegativeBinomialConjugate('
            f'r={self._r}, k={self._k}, n={self._n}, '
            f'alpha={self._alpha}, beta={self._beta})'
        )

    def __eq__(self, other: 'BetaNegativeBinomialConjugate') -> bool:
        return (
            isinstance(other, BetaNegativeBinomialConjugate) and
            self._r == other._r and
            self._k == other._k and
            self._n == other._n and
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )
