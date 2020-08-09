from matplotlib.figure import Figure
from mpl_format.figures import FigureFormatter
from pandas import Series

from probability.distributions import Beta
from probability.distributions.discrete.geometric import Geometric
from probability.distributions.mixins.conjugate import ConjugateMixin
from probability.supports import SUPPORT_BETA


class BetaGeometricConjugate(ConjugateMixin, object):
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
                      distribution. Number of experiments.
        :param beta: Value for the β hyper-parameter of the prior Beta
                     distribution.
        :param n: Number of trials.
        :param k: Number of trials up to and including first success.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._k: int = k

    def prior(self) -> Beta:
        return Beta(
            alpha=self._alpha, beta=self._beta
        ).with_y_label('$P(p=x|α,β)$').prepend_to_label('Prior: ')

    def likelihood(self) -> Geometric:
        return Geometric(p=1 / self._k)

    def posterior(self) -> Beta:
        return Beta(
            alpha=self._alpha + self._n,
            beta=self._beta + self._k
        ).with_y_label('$P(p=x|α+n,β+k)$').prepend_to_label('Posterior: ')

    def plot(self, **kwargs) -> Figure:
        """
        Plot a grid of the different components of the Compound Distribution.

        :param kwargs: kwargs for plot methods
        """
        ff = FigureFormatter(n_rows=2, n_cols=3)
        (
            ax_prior, ax_data, ax_posterior,
            ax_prior_predictive, ax_likelihood, ax_posterior_predictive
        ) = ff.axes.flat

        self.prior().plot(x=SUPPORT_BETA, ax=ax_prior.axes, **kwargs)
        self.posterior().plot(x=SUPPORT_BETA, ax=ax_posterior.axes, **kwargs)

        ax_prior.set_title_text('prior').add_legend()
        ax_posterior.set_title_text('posterior').add_legend()
        ax_prior_predictive.set_title_text('prior predictive').add_legend()
        ax_posterior_predictive.set_title_text(
            'posterior predictive'
        ).add_legend()
        # plot data
        observations = Series(data=[0] * (self._k - 1) + [1])
        observations.index = range(1, self._k + 1)
        observations.plot.bar(ax=ax_data.axes, **kwargs)
        ax_data.set_text(title='data', x_label='i', y_label='$X_i$')
        # plot likelihood
        self.likelihood().plot(k=range(self._n + 1), ax=ax_likelihood.axes)
        ax_likelihood.set_title_text('likelihood')
        return ff.figure

    def __str__(self):

        return f'BetaGeometric(' \
               f'α={self._alpha}, ' \
               f'β={self._beta}, ' \
               f'n={self._n}, ' \
               f'k={self._k}' \
               f')'

    def __repr__(self):

        return f'BetaGeometric(' \
               f'alpha={self._alpha}, ' \
               f'beta={self._beta}, ' \
               f'n={self._n}, ' \
               f'k={self._k}' \
               f')'
