from mpl_format.figures import FigureFormatter
from numpy.ma import arange
from pandas import Series
from scipy.stats import lomax

from probability.distributions.continuous.exponential import Exponential
from probability.distributions.continuous.gamma import Gamma
from probability.distributions.continuous.lomax import Lomax
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin
from probability.distributions.mixins.attributes import AlphaFloatMixin, \
    BetaFloatMixin, NIntMixin


class GammaExponentialConjugate(
    ConjugateMixin,
    PredictiveMixin,
    AlphaFloatMixin, BetaFloatMixin, NIntMixin,
    object
):
    """
    Class for calculating Bayesian probabilities using the Gamma-Exponential
    distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` are the hyper-parameters of the Gamma prior.
    * `α > 0`
    * `β > 0`
    * Interpretation is α prior observations that sum to β.

    Posterior Hyper-parameters
    --------------------------
    * `n` is the number of observations.
    * `x_mean` is the average value (e.g. duration) of x over `n` observations.

    Model parameters
    ----------------
    * `P(x)` is the probability of observing an event p a rate of `x`.
    * `0 ≤ x`

    Links
    -----
    * https://en.wikipedia.org/wiki/Gamma_distribution
    * https://en.wikipedia.org/wiki/Exponential_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior
    """

    def __init__(self, alpha: float, beta: float, n: int, x_mean: float):
        """
        :param alpha: Value for the α hyper-parameter of the prior Gamma
                      distribution (number of observations).
        :param beta: Value for the β hyper-parameter of the prior Gamma
                     distribution (sum of observations).
        :param n: Number of observations.
        :param x_mean: Average duration of, or time between, observations.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._x_mean: float = x_mean
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution = lomax(c=self.alpha_prime,
                                   scale=self.beta_prime)

    @property
    def x_mean(self) -> float:
        return self._x_mean

    @x_mean.setter
    def x_mean(self, value: float):
        self._x_mean = value
        self._reset_distribution()

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._n

    @property
    def beta_prime(self) -> float:
        return self._beta + self._n * self._x_mean

    def prior(self) -> Gamma:
        return Gamma(
            alpha=self._alpha, beta=self._beta
        ).with_y_label('$P(λ=x|α,β)$').prepend_to_label('Prior: ')

    def likelihood(self) -> Exponential:
        return Exponential(lambda_=1 / self._x_mean)

    def posterior(self) -> Gamma:

        return Gamma(
            alpha=self.alpha_prime, beta=self.beta_prime
        ).with_y_label(r'$P(λ=x|α+n,β+n\bar{x})$').prepend_to_label(
            'Posterior: '
        )

    def prior_predictive(self) -> Lomax:

        return Lomax(
            lambda_=self._beta, alpha=self._alpha
        ).with_y_label(r'$P(\tilde{X}=x|α,β)$')

    def posterior_predictive(self) -> Lomax:

        return Lomax(
            lambda_=self.beta_prime,
            alpha=self.alpha_prime
        ).with_y_label(r'$P(\tilde{X}=x|α+n,β+n\bar{x})$')

    def plot(self, **kwargs):
        """
        Plot a grid of the different components of the Compound Distribution.

        :param kwargs: kwargs for plot methods
        """
        ppf_gamma_prior = self.prior().ppf().at(0.99)
        ppf_gamma_posterior = self.posterior().ppf().at(0.99)
        x_gamma_max = int(max(ppf_gamma_prior, ppf_gamma_posterior)) + 1
        x_gamma = arange(0, x_gamma_max + 0.001, 0.001)
        ff = FigureFormatter(n_rows=2, n_cols=3)
        (
            ax_prior, ax_data, ax_posterior,
            ax_prior_predictive, ax_likelihood, ax_posterior_predictive
        ) = ff.axes.flat

        self.prior().plot(x=x_gamma, ax=ax_prior.axes, **kwargs)
        self.posterior().plot(x=x_gamma, ax=ax_posterior.axes, **kwargs)
        y_max_params = max(ax_prior.get_y_max(), ax_posterior.get_y_max())
        ax_prior.set_y_lim(0, y_max_params)
        ax_posterior.set_y_lim(0, y_max_params)

        ppf_lomax_prior = self.prior_predictive().ppf().at(0.99)
        ppf_lomax_posterior = self.posterior_predictive().ppf().at(0.99)
        x_lomax_max = int(max(ppf_lomax_prior, ppf_lomax_posterior)) + 1
        x_lomax = arange(0, x_lomax_max + 0.001, 0.001)
        self.prior_predictive().plot(
            x=x_lomax, kind='line',
            ax=ax_prior_predictive.axes,
            **kwargs
        )
        self.posterior_predictive().plot(
            x=x_lomax, kind='line',
            ax=ax_posterior_predictive.axes,
            **kwargs
        )
        y_max_pred = max(ax_prior_predictive.get_y_max(),
                         ax_posterior_predictive.get_y_max())
        ax_prior_predictive.set_y_lim(0, y_max_pred)
        ax_posterior_predictive.set_y_lim(0, y_max_pred)

        ax_prior.set_title_text('prior').add_legend()
        ax_posterior.set_title_text('posterior').add_legend()
        ax_prior_predictive.set_title_text('prior predictive').add_legend()
        ax_posterior_predictive.set_title_text(
            'posterior predictive'
        ).add_legend()
        # plot data
        observations = Series(
            Exponential(lambda_=1 / self._x_mean).rvs(self._n)
        )
        observations = observations * (1 / self._x_mean) / observations.mean()
        observations.plot.bar(ax=ax_data.axes, **kwargs)
        ax_data.set_text(title='data', x_label='i', y_label='$X_i$')
        # plot likelihood
        x_exponential_max = int(self.likelihood().ppf().at(0.99)) + 1
        x_exponential = arange(0, x_exponential_max + 0.001, 0.001)
        self.likelihood().plot(x=x_exponential, ax=ax_likelihood.axes)
        ax_likelihood.set_title_text('likelihood')
        ax_likelihood.add_legend()
        return ff.figure

    def __str__(self):

        return (
            f'GammaExponential('
            f'α={self._alpha}, '
            f'β={self._beta}, '
            f'n={self._n}, '
            f'x̄={self._x_mean})'
        )

    def __repr__(self):

        return (
            f'GammaExponential('
            f'alpha={self._alpha}, '
            f'beta={self._beta}, '
            f'n={self._n}, '
            f'x_mean={self._x_mean})'
        )
