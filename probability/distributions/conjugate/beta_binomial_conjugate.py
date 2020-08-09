from itertools import product
from typing import Optional, Union, List

from matplotlib.figure import Figure
from mpl_format.figures.figure_formatter import FigureFormatter
from pandas import Series, DataFrame

from probability.distributions.continuous.beta import Beta
from probability.distributions.discrete import Binomial
from probability.distributions.discrete.beta_binomial import BetaBinomial
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin
from probability.supports import SUPPORT_BETA
from probability.utils import is_binomial


class BetaBinomialConjugate(ConjugateMixin,
                            PredictiveMixin,
                            object):
    """
    Class for calculating Bayesian probabilities using the beta-binomial
    distribution.

    The beta-binomial distribution is the binomial distribution in which the
    probability of success at each of n trials is not fixed but randomly drawn
    from a beta distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` are the hyper-parameters of the Beta prior.
    * `α > 0`
    * `β > 0`
    * Interpretation is α-1 successes and β-1 failures.

    Posterior Hyper-parameters
    --------------------------
    * `n` is the number of trials.
    * `m` is the number of successes over `n` trials.

    Model parameters
    ----------------
    * `P(x=1)`, or `θ`, is the probability of a successful trial.
    * `0 ≤ θ ≤ 1`

    Links
    -----
    * https://en.wikipedia.org/wiki/Beta_distribution
    * https://en.wikipedia.org/wiki/Binomial_distribution
    * https://en.wikipedia.org/wiki/Beta-binomial_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_discrete_distribution
    """
    def __init__(self, alpha: float, beta: float, n: int, k: int):
        """
        Create a new beta-binomial distribution.

        :param alpha: Value for the α hyper-parameter of the prior Beta
                      distribution.
        :param beta: Value for the β hyper-parameter of the prior Beta
                      distribution.
        :param n: Observed number of trials.
        :param k: Observed number of successes.
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n
        self._k: int = k

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value

    @property
    def n(self) -> float:
        return self._n

    @n.setter
    def n(self, value: float):
        self._n = value

    @property
    def k(self) -> float:
        return self._k

    @k.setter
    def k(self, value: float):
        self._k = value

    def prior(self) -> Beta:
        return Beta(
            alpha=self._alpha, beta=self._beta
        ).with_y_label('$P(p=x|α,β)$').prepend_to_label('Prior: ')

    def likelihood(self) -> Binomial:
        return Binomial(
            n=self._n,
            p=self._k / self._n
        )

    def posterior(self) -> Beta:
        return Beta(
            alpha=self._alpha + self._k,
            beta=self._beta + self._n - self._k
        ).with_y_label('$P(p=x|α+k,β+n-k)$').prepend_to_label('Posterior: ')

    def prior_predictive(self, n_: Optional[int] = None) -> BetaBinomial:
        """
        Return a BetaBinomial describing the expected distribution of future
        successes of n trials based on the prior parameter estimates.

        :param n_: Optional number of trials over which to calculate probability
                  of success. Leave blank to use the same n as the observed
                  number of trials.
        """
        n_ = n_ if n_ is not None else self._n
        return BetaBinomial(
            n=n_,
            alpha=self._alpha,
            beta=self._beta
        ).with_y_label(r'$P(\tilde{X}=k|\tilde{n},α,β)$')

    def posterior_predictive(self, n_: Optional[int] = None) -> BetaBinomial:
        """
        Return a BetaBinomial describing the expected distribution of future
        successes of n trials based on the posterior parameter estimates.

        :param n_: Optional number of trials over which to calculate probability
                  of success. Leave blank to use the same n as the observed
                  number of trials.
        """
        n_ = n_ if n_ is not None else self._n
        return BetaBinomial(
            n=n_,
            alpha=self._alpha + self._k,
            beta=self._beta + self._n - self._k
        ).with_y_label(r'$P(\tilde{X}=k|\tilde{n},α+k,β+n-k)$')

    def plot(self, n_, **kwargs) -> Figure:
        """
        Plot a grid of the different components of the Compound Distribution.
        
        :param n_: number of trials over which to calculate predictive 
                   distributions
        :param kwargs: kwargs for plot methods
        """
        k_predict = range(n_ + 1)
        ff = FigureFormatter(n_rows=2, n_cols=3)
        (
            ax_prior, ax_data, ax_posterior,
            ax_prior_predictive, ax_likelihood, ax_posterior_predictive
        ) = ff.axes.flat

        self.prior().plot(x=SUPPORT_BETA, ax=ax_prior.axes, **kwargs)
        self.posterior().plot(x=SUPPORT_BETA, ax=ax_posterior.axes, **kwargs)
        self.prior_predictive(n_=n_).plot(
            k=k_predict, kind='bar',
            ax=ax_prior_predictive.axes, **kwargs
        )
        self.posterior_predictive(n_=n_).plot(
            k=k_predict, kind='bar',
            ax=ax_posterior_predictive.axes, **kwargs
        )

        ax_prior.set_title_text('prior').add_legend()
        ax_posterior.set_title_text('posterior').add_legend()
        ax_prior_predictive.set_title_text('prior predictive').add_legend()
        ax_posterior_predictive.set_title_text(
            'posterior predictive'
        ).add_legend()
        # plot data
        observations = Series(
            data=[1] * self._k + [0] * (self._n - self._k)
        ).sample(frac=1)
        observations.index = range(1, self._n + 1)
        observations.plot.bar(ax=ax_data.axes, **kwargs)
        ax_data.set_text(title='data', x_label='i', y_label='$X_i$')
        # plot likelihood
        self.likelihood().plot(k=range(self._n + 1), ax=ax_likelihood.axes)
        ax_likelihood.set_title_text('likelihood')
        return ff.figure

    @staticmethod
    def infer_posterior(data: Series) -> 'Beta':
        """
        Return a new Beta distribution of the posterior most likely to generate
        the given data.

        :param data: Series of `1`s and `0`s or `True`s and `False`s
        """
        alpha: int = data.sum()
        beta: int = len(data) - alpha
        return Beta(alpha=alpha, beta=beta)

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            stats: Optional[Union[str, List[str]]] = None
    ) -> DataFrame:
        """
        Return a DataFrame mapping probability and conditional variables to Beta
        distributions of posteriors most likely to generate the given data.

        :param data: DataFrame containing discrete data.
        :param prob_vars: Name(s) of binomial (or name of multinomial) variables
                          whose posteriors to find probability of.
        :param cond_vars: Names of discrete variables to condition on.
                          Calculations will be done for the cartesian product
                          of variable values
                          e.g if cA={1,2} and cB={3,4} then
                          cAB = {(1,3), (1, 4), (2, 3), (2, 4)}.
        :param stats: Optional stats to append to the output e.g. 'alpha',
                      'median'.
        :return: DataFrame with columns for each conditioning variable, a
                 'prob_var' column indicating the probability variable, a
                 `prob_val` column indicating the value of the probability
                 variable, and a `Beta` column containing the distribution.
        """
        if isinstance(prob_vars, str):
            prob_vars = [prob_vars]
        if len(prob_vars) > 1:
            if not all(is_binomial(data[prob_var]) for prob_var in prob_vars):
                raise ValueError(
                    'If passing more than one prob_var, each must be binomial'
                )
        if isinstance(cond_vars, str):
            cond_vars = [cond_vars]
        cond_products = product(
            *[data[cond_var].unique() for cond_var in cond_vars]
        )
        betas = []
        # iterate over conditions
        for cond_values in cond_products:
            cond_data = data
            cond_dict = {}
            for cond_var, cond_value in zip(cond_vars, cond_values):
                cond_data = cond_data.loc[cond_data[cond_var] == cond_value]
                cond_dict[cond_var] = cond_value
            n_cond: int = len(cond_data)
            for prob_var in prob_vars:
                if is_binomial(data[prob_var]):
                    # one or more binomial columns
                    prob_dict = cond_dict.copy()
                    m_prob: int = cond_data[prob_var].sum()
                    prob_dict['prob_var'] = prob_var
                    prob_dict['prob_val'] = 1
                    prob_dict['Beta'] = Beta(
                        alpha=m_prob, beta=n_cond - m_prob
                    )
                    betas.append(prob_dict)
                else:
                    # single multinomial column
                    for state in data[prob_var].unique():
                        prob_dict = cond_dict.copy()
                        m_prob = len(cond_data.loc[
                            cond_data[prob_var] == state,
                            prob_var
                        ])
                        prob_dict['prob_var'] = prob_var
                        prob_dict['prob_val'] = state
                        prob_dict['Beta'] = Beta(
                            alpha=m_prob, beta=n_cond - m_prob
                        )
                        betas.append(prob_dict)

        betas_data = DataFrame(betas)
        if stats is not None:
            if isinstance(stats, str):
                stats = [stats]
                for stat in stats:
                    if hasattr(Beta, stat):
                        if callable(getattr(Beta, stat)):
                            betas_data[stat] = betas_data['Beta'].map(
                                lambda b: getattr(b, stat)()
                            )
                        else:
                            betas_data[stat] = betas_data['Beta'].map(
                                lambda b: getattr(b, stat)
                            )
        return betas_data

    def __str__(self):

        return (
            f'BetaBinomial('
            f'α={self._alpha}, '
            f'β={self._beta}, '
            f'n={self._n}, k={self._k})'
        )

    def __repr__(self):

        return (
            f'BetaBinomial('
            f'alpha={self._alpha}, '
            f'beta={self._beta}, '
            f'n={self._n}, '
            f'k={self._k})'
        )
