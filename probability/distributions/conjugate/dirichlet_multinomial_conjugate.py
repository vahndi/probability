from itertools import product
from typing import Union, List, Optional, Iterable

from matplotlib.figure import Figure
from mpl_format.figures import FigureFormatter
from pandas import Series, DataFrame
from pandas.core.dtypes.common import is_categorical_dtype

from probability.custom_types.external_custom_types import Array1d, FloatArray1d
from probability.distributions import UniformPrior
from probability.distributions.mixins.conjugate import ConjugateMixin
from probability.distributions.multivariate import Dirichlet, Multinomial
from probability.supports import SUPPORT_DIRICHLET
from probability.utils import num_format


class DirichletMultinomialConjugate(
    ConjugateMixin,
    object
):
    """
    Class for calculating Bayesian probabilities using the dirichlet-multinomial
    distribution.

    The dirichlet-multinomial is a compound probability distribution,
    where a probability vector p is drawn from a Dirichlet distribution with
    parameter vector α, and observations drawn from a multinomial distribution
    with probability vector p and number of trials n.
    The Dirichlet parameter vector α captures the prior belief about the
    situation and can be seen as a pseudo-count of observations of each outcome
    that occur before the actual data is observed.

    Prior Hyper-parameters
    ----------------------
    * `α` is the hyper-parameter vector of the Dirichlet prior, with `K` values.
    * `α1, ..., αK > 0`
    * Interpretation is αk observations of each possible category k in 1, ..., K

    Posterior Hyper-parameters
    --------------------------
    * n is the number of trials. this is the sum of all the
      observations of the different values of k.
    * `x` is the vector of counts of each observed value of k in 1..K.
    * `x = [x1, ..., xK]`
    * `x1, ..., xK >= 0`
    * n = sum(x)

    Model parameters
    ----------------
    * `P(x=k)`, or `p`, is the probability of observing k in a trial.
    * `0 ≤ p ≤ 1`

    Links
    -----
    * https://en.wikipedia.org/wiki/Dirichlet_distribution
    * https://en.wikipedia.org/wiki/Multinomial_distribution
    * https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior
    """
    def __init__(
            self,
            x: Union[FloatArray1d, dict],
            alpha: Union[
                 FloatArray1d, dict, float
             ] = UniformPrior.Dirichlet.alpha
    ):
        """
        Create a new dirichlet-multinomial distribution.

        :param x: Series mapping category names to observed counts.
        :param alpha: Series mapping category names to prior count beliefs.
                      If float, then assumes same prior count to each dimension.
        """
        if isinstance(alpha, Iterable) and isinstance(x, Iterable):
            if len(alpha) != len(x):
                raise ValueError('alpha and x should be the same length')
        if isinstance(x, dict):
            x = Series(x)
        elif not isinstance(x, Series):
            x = Series(
                data=x,
                index=[f'x{k}' for k in range(1, len(x) + 1)]
            )
        if isinstance(alpha, dict):
            alpha = Series(alpha)
        elif not isinstance(alpha, Series):
            alpha = Series(
                data=alpha,
                index=x.index
            )

        self._x = x
        self._n = self._x.sum()
        self._K = len(x)
        self._alpha = alpha

    # region prior hyper-parameters

    @property
    def alpha(self) -> Series:

        return self._alpha

    @alpha.setter
    def alpha(self, value: Array1d):

        if len(value) != self._K:
            raise ValueError(f'alpha must have {self._K} values')
        if not isinstance(value, Series):
            value = Series(
                data=value,
                index=self._alpha.index
            )
        self._alpha = value

    # endregion

    # region posterior hyper-parameters

    @property
    def alpha_prime(self) -> Series:
        return Series({
            self._alpha.index[k]: self._alpha.iloc[k] + self._x.iloc[k]
            for k in range(len(self._alpha))
        })

    # endregion

    @property
    def x(self) -> Series:

        return self._x

    @x.setter
    def x(self, value: Array1d):

        if len(value) != self._K:
            raise ValueError(f'x must have {self._K} values')
        if not isinstance(value, Series):
            value = Series(
                data=value,
                index=self._x.index
            )
        self._x = value

    @property
    def K(self) -> int:
        """
        Number of possible outcomes of any given trial.
        """
        return self._K

    def n(self) -> int:
        """
        Number of trials.
        """
        return self._n

    def prior(self, **kwargs) -> Dirichlet:
        """
        Return a Dirichlet distribution reflecting the prior belief about the
        distribution of the parameter p, before seeing any data.
        """
        return Dirichlet(
            alpha=self._alpha
        ).with_y_label(
            '$P(p_{Mul}=X|α_{Dir})$'
        ).prepend_to_label('Prior: ')

    def likelihood(self, **kwargs) -> Multinomial:
        """
        Return a distribution reflecting the likelihood of observing
        the data, under a Multinomial model, independent of the prior belief
        about the distribution of parameter p.
        """
        return Multinomial(n=self._n, p=self._x / self._n)

    def posterior(self, **kwargs) -> Dirichlet:
        """
        Return a Dirichlet distribution reflecting the posterior belief about
        the distribution of the parameter p, after observing the data.
        """
        return Dirichlet(
            alpha=self.alpha_prime
        ).with_y_label(
            '$P(p_{Mul}=X|'
            'α_{Dir}+x_{Obs})$'
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

        self.prior().plot(x=SUPPORT_DIRICHLET, ax=ax_prior.axes,
                          **kwargs)
        self.posterior().plot(x=SUPPORT_DIRICHLET, ax=ax_posterior.axes,
                              **kwargs)

        ax_prior.set_title_text('prior').add_legend()
        ax_posterior.set_title_text('posterior').add_legend()
        ax_prior_predictive.set_title_text('prior predictive')
        ax_prior_predictive.set_face_color('#bbb')
        ax_posterior_predictive.set_title_text('posterior predictive')
        ax_posterior_predictive.set_face_color('#bbb')
        # plot data
        self._x.plot.bar(ax=ax_data.axes,
                         color=[f'C{i}' for i in range(len(self._x))])
        ax_data.set_text(title='data', x_label='$x_k$',
                         y_label=r'$\vert x_k \vert$')
        # plot likelihood
        likelihood = self.likelihood()
        self.likelihood().plot(k=likelihood.permutations(),
                               ax=ax_likelihood.axes)
        ax_likelihood.set_x_lim(0.5, None)
        ax_likelihood.set_title_text('likelihood')
        ax_likelihood.add_legend()
        return ff.figure

    @staticmethod
    def infer_posterior(
            data: Series,
            alpha: Union[
                FloatArray1d, dict, float
            ] = UniformPrior.Dirichlet.alpha
    ) -> Dirichlet:
        """
        Return a new Dirichlet distribution of the posterior most likely to
        generate the given data.

        :param data: Series of categorical values.
        :param alpha: Value(s) for the α hyper-parameter of the prior Dirichlet
                      distribution.
        """
        if is_categorical_dtype(data):
            categories = data.cat.categories.to_list()
        else:
            categories = list(data.unique())
        x = {
            category: (data == category).sum()
            for category in categories
        }
        return DirichletMultinomialConjugate(x=x, alpha=alpha).posterior()

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            stats: Optional[Union[str, List[str]]] = None,
            alpha: Union[
                FloatArray1d, dict, float
            ] = UniformPrior.Dirichlet.alpha
    ) -> DataFrame:
        """
        Return a DataFrame mapping probability and conditional variables to
        Dirichlet distributions of posteriors most likely to generate the given
        data.

        :param data: DataFrame containing discrete data.
        :param prob_vars: Name(s) of multinomial variables whose posteriors to
                          find probability of.
        :param cond_vars: Names of discrete variables to condition on.
                          Calculations will be done for the cartesian product
                          of variable values
                          e.g if cA = {1, 2} and cB = {3, 4} then
                          cAB = {(1,3), (1, 4), (2, 3), (2, 4)}.
        :param stats: Optional stats to append to the output e.g. 'alpha',
                      'mean'.
        :param alpha: Value(s) for the α hyper-parameter of the prior Dirichlet
                      distribution.
        :return: DataFrame with columns for each conditioning variable, a
                 'prob_var' column indicating the probability variable, and a
                 `Dirichlet` column containing the distribution.
        """
        if isinstance(prob_vars, str):
            prob_vars = [prob_vars]
        if isinstance(cond_vars, str):
            cond_vars = [cond_vars]
        cond_products = product(
            *[data[cond_var].unique() for cond_var in cond_vars]
        )
        dirichlets = []
        # iterate over conditions
        for cond_values in cond_products:
            cond_data = data
            cond_dict = {}
            for cond_var, cond_value in zip(cond_vars, cond_values):
                cond_data = cond_data.loc[cond_data[cond_var] == cond_value]
                cond_dict[cond_var] = cond_value
            for prob_var in prob_vars:
                prob_dict = cond_dict.copy()
                m_prob: Series = cond_data[prob_var].value_counts()
                prob_dict['prob_var'] = prob_var
                prob_dict['Dirichlet'] = DirichletMultinomialConjugate(
                    x=m_prob, alpha=alpha
                ).posterior()
                dirichlets.append(prob_dict)

        dirichlets_data = DataFrame(dirichlets)
        if stats is not None:
            if isinstance(stats, str):
                stats = [stats]
            for stat in stats:
                if hasattr(Dirichlet, stat):
                    if callable(getattr(Dirichlet, stat)):
                        dirichlets_data[stat] = dirichlets_data['Beta'].map(
                            lambda b: getattr(b, stat)()
                        )
                    else:
                        dirichlets_data[stat] = dirichlets_data['Beta'].map(
                            lambda b: getattr(b, stat)
                        )
        return dirichlets_data

    def __str__(self):

        alpha = ', '.join([f'{k}={num_format(v, 3)}'
                           for k, v in self._alpha.items()])
        x = ', '.join([f'{k}={v}' for k, v in self._x.items()])
        return (
            f'DirichletMultinomialConjugate('
            f'α={alpha},'
            f'n={self._n}, '
            f'x={x})'
        )

    def __repr__(self):

        alpha = ', '.join([f'{k}={v}' for k, v in self._alpha.items()])
        x = ', '.join([f'{k}={v}' for k, v in self._x.items()])
        return (
            f'DirichletMultinomialConjugate('
            f'alpha={alpha},'
            f'n={self._n}, '
            f'x={x})'
        )
