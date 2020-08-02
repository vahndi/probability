from typing import Optional

from probability.distributions import Beta
from probability.distributions.discrete.geometric import Geometric
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin


class BetaGeometric(ConjugateMixin):
    """
    Class for calculating Bayesian probabilities using the Shifted
    Beta-Geometric distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` are the hyper-parameters of the Beta prior.
    * `α > 0`
    * `β > 0`
    * Interpretation is α-1 experiments and β-1 failures.

    Posterior Hyper-parameters
    --------------------------
    * `k` is the number of trials until the first success.

    Model parameters
    ----------------
    * `p`, or `θ`, is the probability of a successful trial.
    * `0 ≤ θ ≤ 1`

    Links
    -----
    * https://en.wikipedia.org/wiki/Beta_distribution
    * https://en.wikipedia.org/wiki/Geometric_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_discrete_distribution
    """
    def __init__(self, alpha: float, beta: float, n: int, k: int):
        """
        :param alpha: Value for the α hyper-parameter of the prior Beta
                      distribution.
        :param beta: Value for the β hyper-parameter of the prior Beta
                     distribution.
        :param n: Number of trials.
        :param k: Number of trials until first success.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._k: int = k

    def prior(self) -> Beta:
        return Beta(
            alpha=self._alpha, beta=self._beta
        ).with_x_label('θ').prepend_to_label('Prior: ')

    def likelihood(self, k: Optional[int] = None) -> Geometric:

        k = k if k is not None else self._k
        return Geometric(p=k / self._n).with_x_label('k')

    def posterior(self, k: Optional[int] = None) -> Beta:
        k = k if k is not None else self._k
        return Beta(
            alpha=self._alpha + self._n,
            beta=self._beta + k
        ).with_x_label('p').prepend_to_label('Posterior: ')

    def __str__(self):

        return f'BetaGeometric(' \
               f'n={self._n}, ' \
               f'α={self._alpha}, ' \
               f'β={self._beta})'

    def __repr__(self):

        return f'BetaGeometric(' \
               f'n={self._n}, ' \
               f'alpha={self._alpha}, ' \
               f'beta={self._beta})'
