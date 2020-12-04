from itertools import product
from typing import List, Optional, Union

from mpl_format.figures import FigureFormatter
from numpy.ma import arange
from pandas import Series, DataFrame

from probability.distributions import Poisson, NegativeBinomial
from probability.distributions.conjugate.priors import VaguePrior
from probability.distributions.continuous.gamma import Gamma
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin
from probability.distributions.mixins.attributes import AlphaFloatMixin, \
    BetaFloatMixin, NIntMixin, KIntMixin
from probability.utils import num_format


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

    def __init__(self, n: int, k: int,
                 alpha: float = VaguePrior.Gamma.alpha,
                 beta: float = VaguePrior.Gamma.beta):
        """
        :param n: Number of intervals.
        :param k: Number of occurrences.
        :param alpha: Value for the α hyper-parameter of the prior Gamma
                      distribution (number of occurrences).
        :param beta: Value for the β hyper-parameter of the prior Gamma
                     distribution (number of intervals).
        """
        self._n: int = n
        self._k: k = k
        self._alpha: float = alpha
        self._beta: float = beta

    # region posterior hyper-parameters

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._k

    @property
    def beta_prime(self) -> float:
        return self._beta + self._n

    # endregion

    def prior(self) -> Gamma:
        """
        Return a Gamma distribution reflecting the prior belief about the
        distribution of the parameter λ, before seeing any data.
        """
        return Gamma(
            alpha=self._alpha, beta=self._beta
        ).with_y_label(
            '$P(λ_{Poi}=x|'
            'α_{Gam},'
            'β_{Gam})$'
        ).prepend_to_label('Prior: ')

    def likelihood(self) -> Poisson:
        """
        Return a distribution reflecting the likelihood of observing
        the data, under a Poisson model, independent of the prior belief
        about the distribution of parameter λ.
        """
        return Poisson(lambda_=self._k / self._n)

    def posterior(self) -> Gamma:
        """
        Return a Gamma distribution reflecting the posterior belief about the
        distribution of the parameter λ, after observing the data.
        """
        return Gamma(
            alpha=self.alpha_prime, beta=self.beta_prime
        ).with_y_label(
            r'$P(λ_{Poi}=x|'
            r'α_{Gam}+k_{Obs},'
            r'β_{Gam}+n_{Obs})$'
        ).prepend_to_label(
            'Posterior: '
        )

    # region predictive

    def prior_predictive(self) -> NegativeBinomial:

        return NegativeBinomial(
            r=self._alpha,
            p=1 / (1 + self._beta)
        ).with_y_label(
            r'$P(\tilde{X}=x|'
            r'α_{Gam},'
            r'β_{Gam})$'
        )

    def posterior_predictive(self) -> NegativeBinomial:

        return NegativeBinomial(
            r=self.alpha_prime,
            p=1 / (1 + self.beta_prime)
        ).with_y_label(
            r'$P(\tilde{X}=x|'
            r'α_{Gam}+k_{Obs},'
            r'β_{Gam}+n_{Obs})$'
        )

    # endregion

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

    @staticmethod
    def infer_posterior(data: Series,
                        alpha: float = VaguePrior.Gamma.alpha,
                        beta: float = VaguePrior.Gamma.beta) -> Gamma:
        """
        Return a new Gamma distribution of the posterior most likely to generate
        the given data.

        :param data: Series of integers representing the number of occurrences
                     per interval.
        :param alpha: Value for the α hyper-parameter of the prior Gamma
                      distribution (number of occurrences).
        :param beta: Value for the β hyper-parameter of the prior Gamma
                     distribution (number of intervals).
        """
        k: int = data.sum()
        n: int = len(data)
        return GammaPoissonConjugate(
            n=n, k=k, alpha=alpha, beta=beta
        ).posterior()

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            alpha: float = VaguePrior.Gamma.alpha,
            beta: float = VaguePrior.Gamma.beta,
            stats: Optional[Union[str, dict, List[Union[str, dict]]]] = None
    ) -> DataFrame:
        """
        Return a DataFrame mapping probability and conditional variables to Beta
        distributions of posteriors most likely to generate the given data.

        :param data: DataFrame containing discrete data.
        :param prob_vars: Name(s) of poisson variables whose posteriors to
                          find probability of.
        :param cond_vars: Names of discrete variables to condition on.
                          Calculations will be done for the cartesian product
                          of variable values
                          e.g if cA={1,2} and cB={3,4} then
                          cAB = {(1,3), (1, 4), (2, 3), (2, 4)}.
        :param alpha: Value for the α hyper-parameter of each prior Gamma
                      distribution (number of occurrences).
        :param beta: Value for the β hyper-parameter of each prior Gamma
                     distribution (number of intervals).
        :param stats: Optional stats to append to the output e.g. 'alpha',
                      'median'. To pass arguments use a dict mapping stat
                      name to iterable of args.
        :return: DataFrame with columns for each conditioning variable, a
                 'prob_var' column indicating the probability variable, a
                 `prob_val` column indicating the value of the probability
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
            for prob_var in prob_vars:
                prob_dict = cond_dict.copy()
                prob_dict['prob_var'] = prob_var
                gamma = GammaPoissonConjugate.infer_posterior(
                    data=cond_data[prob_var],
                    alpha=alpha, beta=beta
                )
                prob_dict['Gamma'] = gamma
                for stat in stats:
                    prob_dict = {**prob_dict, **gamma.stat(stat, True)}
                gammas.append(prob_dict)

        gammas_data = DataFrame(gammas)

        return gammas_data

    def __str__(self):

        return (
            f'GammaPoissonConjugate('
            f'α={num_format(self._alpha, 3)}, '
            f'β={num_format(self._beta, 3)}, '
            f'n={self._n}, '
            f'k={self._k})'
        )

    def __repr__(self):

        return (
            f'GammaPoissonConjugate('
            f'alpha={self._alpha}, '
            f'beta={self._beta}, '
            f'n={self._n}, '
            f'k={self._k})'
        )
