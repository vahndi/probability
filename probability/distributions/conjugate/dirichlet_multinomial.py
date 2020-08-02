from itertools import product
from typing import Union, List, Optional

from pandas import Series, DataFrame
from pandas.core.dtypes.common import is_categorical_dtype

from probability.custom_types import Array1d, FloatArray1d
from probability.distributions.multivariate import Dirichlet, Multinomial
from probability.distributions.mixins.conjugate_mixin import ConjugateMixin


class DirichletMultinomial(ConjugateMixin):
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
    * `x1, ...xK >= 0`
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
    * https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_discrete_distribution
    """
    def __init__(self,
                 alpha: Union[FloatArray1d, dict],
                 n: int,
                 x: Union[FloatArray1d, dict]):
        """
        Create a new dirichlet-multinomial distribution.

        :param alpha: Series mapping category names to prior count beliefs.
        :param n: Number of trials.
        :param x: Series mapping category names to observed counts.
        """
        if type(alpha) != type(x):
            raise TypeError('alpha and x should the type of array')
        if len(alpha) != len(x):
            raise ValueError('alpha and x should be the same length')
        if not isinstance(alpha, Series):
            alpha = Series(
                data=alpha,
                index=[f'α{k}' for k in range(1, len(alpha) + 1)]
            )
            x = Series(
                data=x,
                index=[f'x{k}' for k in range(1, len(x) + 1)]
            )

        self._alpha = alpha
        self._x = x
        self._n = n
        self._K = len(x)

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

    def prior(self, **kwargs) -> Dirichlet:

        return Dirichlet(
            alpha=self._alpha
        )

    def likelihood(self, **kwargs) -> Multinomial:

        return Multinomial(n=self._n, p=self._x)

    def posterior(self, **kwargs) -> Dirichlet:

        return Dirichlet(
            alpha=Series({
                self._alpha.index[k]: self._alpha.iloc[k] + self._x.iloc[k]
                for k in range(len(self._alpha))
            })
        )

    @staticmethod
    def infer_posterior(data: Series) -> Dirichlet:
        """
        Return a new Dirichlet distribution of the posterior most likely to
        generate the given data.

        :param data: Series of categorical values.
        """
        if is_categorical_dtype(data):
            categories = data.cat.categories.to_list()
        else:
            categories = list(data.unique())
        alpha = {
            category: (data == category).sum()
            for category in categories
        }
        return Dirichlet(alpha=alpha)

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            stats: Optional[Union[str, List[str]]] = None
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
                          e.g if cA={1,2} and cB={3,4} then
                          cAB = {(1,3), (1, 4), (2, 3), (2, 4)}.
        :param stats: Optional stats to append to the output e.g. 'alpha',
                      'mean'.
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
                m_prob: int = cond_data[prob_var].value_counts()
                prob_dict['prob_var'] = prob_var
                prob_dict['Dirichlet'] = Dirichlet(alpha=m_prob)
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

        alpha = ', '.join([f'{k}={v}' for k, v in self._alpha.items()])
        x = ', '.join([f'{k}={v}' for k, v in self._x.items()])
        return (
            f'DirichletMultinomial('
            f'α={alpha},'
            f'n={self._n}, '
            f'x={x})'
        )

    def __repr__(self):

        alpha = ', '.join([f'{k}={v}' for k, v in self._alpha.items()])
        x = ', '.join([f'{k}={v}' for k, v in self._x.items()])
        return (
            f'DirichletMultinomial('
            f'alpha={alpha},'
            f'n={self._n}, '
            f'x={x})'
        )
