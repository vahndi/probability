from itertools import product
from typing import Union, List, Optional

from matplotlib.figure import Figure
from mpl_format.figures import FigureFormatter
from numpy.random import choice
from pandas import Series, DataFrame

from probability.distributions.conjugate.priors import UniformPrior
from probability.distributions.continuous.beta import Beta
from probability.distributions.discrete.geometric import Geometric
from probability.distributions.mixins.attributes import AlphaFloatMixin, \
    BetaFloatMixin, NIntMixin, KIntMixin
from probability.distributions.mixins.conjugate import ConjugateMixin
from probability.supports import SUPPORT_BETA
from probability.utils import is_binary, num_format


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
      This is equal to the total number of trials across all experiments.
    * `n` is the number of experiments (each experiment consists of one or more
                                        trials, each ending with a success)

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
    def __init__(self, n: int, k: int,
                 alpha: float = UniformPrior.Geometric.alpha,
                 beta: float = UniformPrior.Geometric.beta):
        """
        :param n: Number of experiments.
        :param k: Number of trials up to and including each success.
        :param alpha: Value for the α hyper-parameter of the prior Beta
                      distribution. Number of prior experiments.
        :param beta: Value for the β hyper-parameter of the prior Beta
                     distribution. Number of prior failures
                     ( = total trials - successful trials).
        """
        self._n: int = n
        self._k: int = k
        self._alpha: float = alpha
        self._beta: float = beta

    # region posterior hyper-parameters

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._n

    @property
    def beta_prime(self) -> float:
        return self._beta + self._k - self._n

    # endregion

    def prior(self) -> Beta:
        """
        Return a Beta distribution reflecting the prior belief about the
        distribution of the parameter p, before seeing any data.
        """
        return Beta(
            alpha=self._alpha, beta=self._beta
        ).with_y_label(
            '$P(p_{Geom}=x|'
            'α_{Beta},'
            'β_{Beta})$'
        ).prepend_to_label('Prior: ')

    def likelihood(self) -> Geometric:
        """
        Return a distribution reflecting the likelihood of observing
        the data, under a Geometric model, independent of the prior belief about
        the distribution of parameter p.
        """
        return Geometric(p=self._n / self._k)

    def posterior(self) -> Beta:
        """
        Return a Beta distribution reflecting the posterior belief about the
        distribution of the parameter p, after observing the data.
        """
        return Beta(
            alpha=self.alpha_prime,
            beta=self.beta_prime
        ).with_y_label(
            '$P(p_{Geom}=x|'
            'α_{Beta}+n_{Obs},'
            'β_{Beta}+k_{Obs}-n_{Obs})$'
        ).prepend_to_label('Posterior: ')

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
        ax_prior_predictive.set_title_text('prior predictive')
        ax_prior_predictive.set_face_color('#bbb')
        ax_posterior_predictive.set_title_text(
            'posterior predictive'
        )
        ax_posterior_predictive.set_face_color('#bbb')
        # plot data
        observations = Series(data=[0] * (self._k - 1) + [1])
        i_success = choice(a=range(self._k - 1),
                           size=self._n - 1, replace=False)
        observations.iloc[i_success] = 1
        observations.index = range(1, self._k + 1)
        observations.plot.bar(ax=ax_data.axes, **kwargs)
        ax_data.set_text(title='data', x_label='i', y_label='$X_i$')
        # plot likelihood
        self.likelihood().plot(k=range(self._n + 1), ax=ax_likelihood.axes)
        ax_likelihood.set_x_lim(0.5, None)
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

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            stats: Optional[Union[str, dict, List[Union[str, dict]]]] = None,
            alpha: float = UniformPrior.Geometric.alpha,
            beta: float = UniformPrior.Geometric.beta
    ) -> DataFrame:
        """
        Return a DataFrame mapping probability and conditional variables to Beta
        distributions of posteriors most likely to generate the given data.

        :param data: DataFrame containing discrete data.
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
        :param alpha: Value for the α hyper-parameter of each prior Beta
                      distribution.
        :param beta: Value for the β hyper-parameter of each prior Beta
                     distribution.
        :return: DataFrame with columns for each conditioning variable,
                 a 'prob_var' column indicating the probability variable,
                 a `prob_val` column indicating the value of the probability
                 variable, and a `Beta` column containing the distribution.
        """
        if isinstance(prob_vars, str):
            prob_vars = [prob_vars]
        if not all(is_binary(data[prob_var]) for prob_var in prob_vars):
            raise ValueError('Prob vars must be binary valued')
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
        geoms = []
        # iterate over conditions
        for cond_values in cond_products:
            cond_data = data
            cond_dict = {}
            for cond_var, cond_value in zip(cond_vars, cond_values):
                cond_data = cond_data.loc[cond_data[cond_var] == cond_value]
                cond_dict[cond_var] = cond_value
            for prob_var in prob_vars:
                # one or more binomial columns
                prob_dict = cond_dict.copy()
                prob_dict['prob_var'] = prob_var
                prob_dict['prob_val'] = 1
                posterior = BetaGeometricConjugate.infer_posterior(
                    alpha=alpha, beta=beta,
                    data=cond_data[prob_var]
                )
                prob_dict['Beta'] = posterior
                for stat in stats:
                    prob_dict = {**prob_dict,
                                 ** posterior.stat(stat, True)}
                geoms.append(prob_dict)

        geoms_data = DataFrame(geoms)

        return geoms_data

    def __str__(self):

        return f'BetaGeometricConjugate(' \
               f'α={num_format(self._alpha, 3)}, ' \
               f'β={num_format(self._beta, 3)}, ' \
               f'n={self._n}, ' \
               f'k={self._k}' \
               f')'

    def __repr__(self):

        return f'BetaGeometricConjugate(' \
               f'alpha={self._alpha}, ' \
               f'beta={self._beta}, ' \
               f'n={self._n}, ' \
               f'k={self._k}' \
               f')'
