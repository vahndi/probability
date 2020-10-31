from matplotlib.figure import Figure
from mpl_format.figures import FigureFormatter
from pandas import Series

from probability.distributions import Beta
from probability.distributions.conjugate.priors import UniformPrior
from probability.distributions.discrete.geometric import Geometric
from probability.distributions.mixins.conjugate import ConjugateMixin
from probability.distributions.mixins.attributes import AlphaFloatMixin, \
    BetaFloatMixin, NIntMixin, KIntMixin
from probability.supports import SUPPORT_BETA


class BetaGeometricConjugate(
    ConjugateMixin,
    AlphaFloatMixin, BetaFloatMixin, NIntMixin, KIntMixin,
    object
):
    """
    Class for calculating Bayesian probabilities using the Shifted
    Beta-Geometric distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` are the hyper-parameters of the Beta prior.
    * `α > 0`
    * `β > 0`
    * Interpretation is α experiments and β failures.

    Posterior Hyper-parameters
    --------------------------
    * `k` is the number of trials up to and including the first success.
      This is equal to the total number of observations across all experiments.
    * `n` is the number of experiments (each experiment consists of one or more
                                        trials)

    Model parameters
    ----------------
    * `p`, or `θ`, is the probability of a successful trial.
    * `0 ≤ θ ≤ 1`

    Links
    -----
    * https://en.wikipedia.org/wiki/Beta_distribution
    * https://en.wikipedia.org/wiki/Geometric_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior
    """
    def __init__(self, alpha: float, beta: float, n: int, k: int):
        """
        :param alpha: Value for the α hyper-parameter of the prior Beta
                      distribution. Number of experiments.
        :param beta: Value for the β hyper-parameter of the prior Beta
                     distribution.
        :param n: Number of experiments.
        :param k: Number of trials up to and including first success.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._k: int = k

    # region posterior hyper-parameters

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._n

    @property
    def beta_prime(self) -> float:
        return self._beta + self._k - self._n

    # endregion

    def prior(self) -> Beta:
        return Beta(
            alpha=self._alpha, beta=self._beta
        ).with_y_label('$P(p=x|α,β)$').prepend_to_label('Prior: ')

    def likelihood(self) -> Geometric:
        return Geometric(p=1 / self._k)

    def posterior(self) -> Beta:
        return Beta(
            alpha=self.alpha_prime,
            beta=self.beta_prime
        ).with_y_label('$P(p=x|α+n,β+k-n)$').prepend_to_label('Posterior: ')

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
        ax_likelihood.add_legend()
        return ff.figure

    @staticmethod
    def infer_posterior(data: Series,
                        alpha: float = UniformPrior.Geometric.alpha,
                        beta: float = UniformPrior.Geometric.beta) -> Beta:
        """
        Return a new Beta distribution of the posterior most likely to generate
        the given data.

        Assumes that each experiment was completed such that the number of
        # successes observed equals the number of experiments.

        https://en.wikipedia.org/wiki/Geometric_distribution#Statistical_inference

        :param data: Series of `1`s and `0`s or `True`s and `False`s
                     The data represents a series of experiments where each
                     experiment ends with an observation of 1 preceded by
                     0 or more observations of 0.
        :param alpha: Value for the α hyper-parameter of the prior Beta
                      distribution.
        :param beta: Value for the β hyper-parameter of the prior Beta
                     distribution.
        """
        # assume that each experiment was completed such that the number of
        # successes observed equals the number of experiments
        num_experiments = data.sum()
        num_trials = len(data)
        return BetaGeometricConjugate(
            alpha=alpha, beta=beta,
            n=num_experiments,
            k=num_trials
        ).posterior()

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
