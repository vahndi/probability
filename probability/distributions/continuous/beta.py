from itertools import product

from pandas import DataFrame, Series
from scipy.stats import beta as beta_dist, rv_continuous
from typing import List, Union, Optional

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin
from probability.distributions.special import prob_bb_greater_exact


class Beta(RVContinuous1dMixin):

    def __init__(self, alpha: float, beta: float):

        self._alpha: float = alpha
        self._beta: float = beta
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = beta_dist(self._alpha, self._beta)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self._reset_distribution()

    @staticmethod
    def infer(data: Series) -> 'Beta':
        """
        Return a new Beta distribution using a Series for counts.

        :param data: Series of `1`s and `0`s or `True`s and `False`s
        """
        alpha: int = data.sum()
        beta: int = len(data) - alpha
        return Beta(alpha=alpha, beta=beta)

    @staticmethod
    def infers(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            stats: Optional[Union[str, List[str]]] = None
    ) -> DataFrame:
        """
        Return a DataFrame mapping probability and conditional variables to Beta
        distributions.

        :param data: DataFrame containing discrete data.
        :param prob_vars: Names of binary variables to find probability of.
        :param cond_vars: Names of discrete variables to condition on.
                          Calculations will be done for the cartesian product
                          of values in each variable.
        :param stats: Optional stats to append to the output e.g. 'alpha',
                      'median'.
        :return: DataFrame with columns for each conditioning variable, a 'p'
                 column indicating the probability variable and a 'Beta'
                 column containing the distribution.
        """
        if isinstance(prob_vars, str):
            prob_vars = [prob_vars]
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
                prob_dict = cond_dict.copy()
                m_prob: int = cond_data[prob_var].sum()
                prob_dict['p'] = prob_var
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

        return f'Beta(α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'

    def __gt__(self, other: 'Beta') -> float:

        return prob_bb_greater_exact(
            alpha_1=self._alpha, beta_1=self._beta, m_1=0, n_1=0,
            alpha_2=other._alpha, beta_2=other._beta, m_2=0, n_2=0
        )

    def __lt__(self, other: 'Beta') -> float:

        return other < self
