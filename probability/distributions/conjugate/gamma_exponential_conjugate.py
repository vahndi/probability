from itertools import product
from typing import Union, List, Optional

from mpl_format.figures import FigureFormatter
from numpy.ma import arange
from pandas import Series, DataFrame

from probability.distributions.conjugate.priors import VaguePrior
from probability.distributions.continuous.exponential import Exponential
from probability.distributions.continuous.gamma import Gamma
from probability.distributions.continuous.lomax import Lomax
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin
from probability.distributions.mixins.attributes import AlphaFloatMixin, \
    BetaFloatMixin, NIntMixin
from probability.utils import num_format


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
    def __init__(self, n: int, x_mean: float,
                 alpha: float = VaguePrior.Gamma.alpha,
                 beta: float = VaguePrior.Gamma.beta):
        """
        :param n: Number of observations.
        :param x_mean: Average duration of, or time between, observations.
        :param alpha: Value for the α hyper-parameter of the prior Gamma
                      distribution (number of observations).
        :param beta: Value for the β hyper-parameter of the prior Gamma
                     distribution (sum of observations).
        """
        self._n: int = n
        self._x_mean: float = x_mean
        self._alpha: float = alpha
        self._beta: float = beta

    @property
    def x_mean(self) -> float:
        return self._x_mean

    @x_mean.setter
    def x_mean(self, value: float):
        self._x_mean = value

    # region posterior hyper-parameters

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._n

    @property
    def beta_prime(self) -> float:
        return self._beta + self._n * self._x_mean

    # endregion

    def prior(self) -> Gamma:
        """
        Return a Gamma distribution reflecting the prior belief about the
        distribution of the parameter λ, before seeing any data.
        """
        return Gamma(
            alpha=self._alpha, beta=self._beta
        ).with_y_label(
            '$P(λ_{Exp}=x|'
            'α_{Gam},'
            'β_{Gam})$'
        ).prepend_to_label('Prior: ')

    def likelihood(self) -> Exponential:
        """
        Return a distribution reflecting the likelihood of observing
        the data, under an Exponential model, independent of the prior belief
        about the distribution of parameter λ.
        """
        return Exponential(lambda_=1 / self._x_mean)

    def posterior(self) -> Gamma:
        """
        Return a Gamma distribution reflecting the posterior belief about the
        distribution of the parameter λ, after observing the data.
        """
        return Gamma(
            alpha=self.alpha_prime, beta=self.beta_prime
        ).with_y_label(
            r'$P(λ_{Exp}=x|'
            r'α_{Gam}+n_{Obs},'
            r'β_{Gam}+n_{Obs}\bar{x}_{Obs})$'
        ).prepend_to_label(
            'Posterior: '
        )

    # region predictive

    def prior_predictive(self) -> Lomax:

        return Lomax(
            lambda_=self._beta, alpha=self._alpha
        ).with_y_label(
            r'$P(\tilde{X}=x|'
            r'α_{Gam},'
            r'β_{Gam})$'
        )

    def posterior_predictive(self) -> Lomax:

        return Lomax(
            lambda_=self.beta_prime,
            alpha=self.alpha_prime
        ).with_y_label(
            r'$P(\tilde{X}=x|'
            r'α_{Gam}+n_{Obs},'
            r'β_{Gam}+n_{Obs}\bar{x}_{Obs})$'
        )

    # endregion

    @staticmethod
    def infer_posterior(data: Series,
                        alpha: float = VaguePrior.Gamma.alpha,
                        beta: float = VaguePrior.Gamma.beta) -> Gamma:
        """
        Return a new Gamma distribution of the posterior most likely to
        generate the given data.

        :param data: Series of float values representing duration of,
                     or between each observation.
        :param alpha: Value for the α hyper-parameter of the prior Gamma
                      distribution (number of observations).
        :param beta: Value for the β hyper-parameter of the prior Gamma
                     distribution (sum of observations).
        """
        n = len(data)
        x_mean = data.mean()
        return GammaExponentialConjugate(
            n=n, x_mean=x_mean,
            alpha=alpha, beta=beta
        ).posterior()

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            stats: Optional[Union[str, dict, List[Union[str, dict]]]] = None,
            alpha: float = VaguePrior.Gamma.alpha,
            beta: float = VaguePrior.Gamma.beta
    ) -> DataFrame:
        """
        Return a DataFrame mapping probability and conditional variables to Beta
        distributions of posteriors most likely to generate the given data.

        :param data: DataFrame containing observation data.
        :param prob_vars: Name(s) of binary variables
                          whose posteriors to find probability of.
        :param cond_vars: Names of discrete variables to condition on.
                          Calculations will be done for the cartesian product
                          of variable values
                          e.g if cA={1,2} and cB={3,4} then
                          cAB = {(1,3), (1, 4), (2, 3), (2, 4)}.
        :param stats: Optional stats to append to the output e.g. 'alpha',
                      'median'. To pass arguments use a dict mapping stat
                      name to iterable of args.
        :param alpha: Value for the α hyper-parameter of each prior Gamma
                      distribution (number of observations).
        :param beta: Value for the β hyper-parameter of each prior Gamma
                     distribution (sum of observations).
        :return: DataFrame with columns for each conditioning variable,
                 a 'prob_var' column indicating the probability variable,
                 a `prob_val` column indicating the value of the probability
                 variable, and a `Beta` column containing the distribution.
        """
        if isinstance(prob_vars, str):
            prob_vars = [prob_vars]
        if isinstance(cond_vars, str):
            cond_vars = [cond_vars]
        cond_products = product(
            *[data[cond_var].unique() for cond_var in cond_vars]
        )
        if stats is not None:
            if isinstance(stats, str) or isinstance(stats, dict):
                stats = [stats]
        else:
            stats = []
        gammas = []
        # iterate over conditions
        for cond_values in cond_products:
            cond_data = data
            cond_dict = {}
            for cond_var, cond_value in zip(cond_vars, cond_values):
                cond_data = cond_data.loc[cond_data[cond_var] == cond_value]
                cond_dict[cond_var] = cond_value
            n_cond: int = len(cond_data)
            for prob_var in prob_vars:
                prob_dict = cond_dict.copy()
                x_prob: float = cond_data[prob_var].mean()
                prob_dict['prob_var'] = prob_var
                posterior = GammaExponentialConjugate(
                    n=n_cond, x_mean=x_prob,
                    alpha=alpha, beta=beta
                ).posterior()
                prob_dict['Gamma'] = posterior
                for stat in stats:
                    prob_dict = {**prob_dict,
                                 ** posterior.stat(stat, True)}
                gammas.append(prob_dict)

        gammas_data = DataFrame(gammas)

        return gammas_data

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
            f'GammaExponentialConjugate('
            f'n={self._n}, '
            f'x̄={self._x_mean})'
            f'α={num_format(self._alpha, 3)}, '
            f'β={num_format(self._beta, 3)}, '
        )

    def __repr__(self):

        return (
            f'GammaExponentialConjugate('
            f'n={self._n}, '
            f'x_mean={self._x_mean})'
            f'alpha={self._alpha}, '
            f'beta={self._beta}, '
        )
