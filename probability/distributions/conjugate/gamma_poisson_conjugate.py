from mpl_format.figures import FigureFormatter
from numpy.ma import arange
from pandas import Series

from probability.distributions import Poisson, NegativeBinomial
from probability.distributions.continuous.gamma import Gamma
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin, AlphaFloatMixin, BetaFloatMixin, NIntMixin, KIntMixin


class GammaPoissonConjugate(
    ConjugateMixin,
    PredictiveMixin,
    AlphaFloatMixin, BetaFloatMixin, NIntMixin, KIntMixin,
    object
):
    """
    Class for calculating Bayesian probabilities using the Gamma-Poisson
    distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` are the hyper-parameters of the Gamma prior.
    * `α > 0`
    * `β > 0`
    * Interpretation is α total occurrences in β intervals.

    Posterior Hyper-parameters
    --------------------------
    * `n` is the number of intervals.
    * `k` is the number of occurrences.

    Model parameters
    ----------------
    * `P(k)` is the probability of observing k events in an interval.
    * `k ≥ 0`

    Links
    -----
    * https://en.wikipedia.org/wiki/Gamma_distribution
    * https://en.wikipedia.org/wiki/Poisson_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior
    """

    def __init__(self, alpha: float, beta: float, n: int, k: int):
        """
        :param alpha: Value for the α hyper-parameter of the prior Gamma
                      distribution (number of occurrences).
        :param beta: Value for the β hyper-parameter of the prior Gamma
                     distribution (number of intervals).
        :param n: Number of intervals.
        :param k: Number of occurrences.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._k: k = k

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._k

    @property
    def beta_prime(self) -> float:
        return self._beta + self._n

    def prior(self) -> Gamma:
        return Gamma(
            alpha=self._alpha, beta=self._beta
        ).with_y_label('$P(λ=x|α,β)$').prepend_to_label('Prior: ')

    def likelihood(self) -> Poisson:
        return Poisson(lambda_=self._k / self._n)

    def posterior(self) -> Gamma:

        return Gamma(
            alpha=self.alpha_prime, beta=self.beta_prime
        ).with_y_label(r'$P(λ=x|α+k,β+n)$').prepend_to_label(
            'Posterior: '
        )

    def prior_predictive(self) -> NegativeBinomial:

        return NegativeBinomial(
            r=self._alpha,
            p=1 / (1 + self._beta)
        ).with_y_label(r'$P(\tilde{X}=x|α,β)$')

    def posterior_predictive(self) -> NegativeBinomial:

        return NegativeBinomial(
            r=self.alpha_prime,
            p=1 / (1 + self.beta_prime)
        ).with_y_label(r'$P(\tilde{X}=x|α+k,β+n)$')

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
        ppf_n_binom_prior = self.prior_predictive().ppf().at(0.99)
        ppf_n_binom_posterior = self.prior_predictive().ppf().at(0.99)
        k_pred = range(int(max(ppf_n_binom_prior, ppf_n_binom_posterior)) + 1)
        self.prior_predictive().plot(
            k=k_pred,
            ax=ax_prior_predictive.axes,
            **kwargs
        )
        self.posterior_predictive().plot(
            k=k_pred,
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
        observations = Series(self.likelihood().rvs(self._n))
        observations.plot.bar(ax=ax_data.axes, **kwargs)
        ax_data.set_text(title='data', x_label='i', y_label='$X_i$')
        # plot likelihood
        k_poisson = range(int(self.likelihood().ppf().at(0.99)) + 2)
        self.likelihood().plot(k=k_poisson, ax=ax_likelihood.axes)
        ax_likelihood.set_title_text('likelihood')
        ax_likelihood.add_legend()
        return ff.figure

    def __str__(self):

        return (
            f'GammaExponential('
            f'α={self._alpha}, '
            f'β={self._beta}, '
            f'n={self._n}, '
            f'k={self._k})'
        )

    def __repr__(self):

        return (
            f'GammaExponential('
            f'alpha={self._alpha}, '
            f'beta={self._beta}, '
            f'n={self._n}, '
            f'k={self._k})'
        )
