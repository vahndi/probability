from itertools import product
from math import sqrt
from typing import Optional, Union, List

from matplotlib.figure import Figure
import numpy as np
from numpy import array, ndarray, power
from pandas import Series, DataFrame

from probability.custom_types.external_custom_types import Array1d
from probability.distributions.conjugate.priors import VaguePrior
from probability.distributions.continuous.gamma import Gamma
from probability.distributions.continuous.students_t import StudentsT
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin
from probability.plotting import new_subplots
from probability.utils import num_format


class GammaNormalConjugate(
    ConjugateMixin,
    PredictiveMixin,
    object
):
    """
    Bayesian Normal model with known mean μ and unknown precision τ ~ Gamma(α, β).

    Likelihood: x_i ~ Normal(μ, σ² = 1/τ)
    Prior:      τ ~ Gamma(α, β)
    Posterior:  τ | x ~ Gamma(α', β')
                where α' = α + n / 2,
                      β' = β + 1/2 * sum((x_i - μ)²)
    Predictive: x_new | x ~ Student's t(ν = 2α', μ = μ, σ = sqrt(β' / α'))
    """

    def __init__(
            self,
            mu: float,
            alpha: float = VaguePrior.Gamma.alpha,
            beta: float = VaguePrior.Gamma.beta,
            x: Optional[Array1d] = None,
            n: Optional[int] = None,
            sum_sq_diff: Optional[float] = None
    ):
        """
        Create a new Gamma-Normal conjugate distribution.

        :param mu: Known mean of the Normal data-generating distribution.
        :param alpha: Shape parameter of the Gamma prior on precision τ.
        :param beta: Rate parameter of the Gamma prior on precision τ.
        :param x: Array or Series of observed data.
        :param n: Number of observations (if x not directly passed).
        :param sum_sq_diff: Sum of squared differences sum((x_i - mu)**2).
        """
        self._mu: float = float(mu)
        self._alpha: float = float(alpha)
        self._beta: float = float(beta)

        if x is not None:
            self._x: Optional[ndarray] = array(x)
            self._n: int = len(self._x)
            self._sum_sq_diff: float = float(power(self._x - self._mu, 2).sum())
        elif n is not None:
            self._x: Optional[ndarray] = None
            self._n: int = int(n)
            self._sum_sq_diff: float = (
                float(sum_sq_diff) if sum_sq_diff is not None else 0.0
            )
        else:
            self._x: Optional[ndarray] = None
            self._n: int = 0
            self._sum_sq_diff: float = 0.0

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = float(value)
        if self._x is not None:
            self._sum_sq_diff = float(power(self._x - self._mu, 2).sum())

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = float(value)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = float(value)

    @property
    def x(self) -> Optional[ndarray]:
        return self._x

    @x.setter
    def x(self, value: Array1d):
        self._x = array(value)
        self._n = len(self._x)
        self._sum_sq_diff = float(power(self._x - self._mu, 2).sum())

    @property
    def n(self) -> int:
        return self._n

    @property
    def sum_sq_diff(self) -> float:
        return self._sum_sq_diff

    @property
    def alpha_prime(self) -> float:
        return self._alpha + self._n / 2.0

    @property
    def beta_prime(self) -> float:
        return self._beta + self._sum_sq_diff / 2.0

    def prior(self) -> Gamma:
        return Gamma(
            alpha=self._alpha, beta=self._beta
        ).with_x_label('τ').prepend_to_label('Prior: ')

    def posterior(self) -> Gamma:
        return Gamma(
            alpha=self.alpha_prime, beta=self.beta_prime
        ).with_x_label('τ').prepend_to_label('Posterior: ')

    def prior_predictive(self) -> StudentsT:
        return StudentsT(
            nu=2.0 * self._alpha,
            mu=self._mu,
            sigma=sqrt(self._beta / self._alpha)
        ).with_x_label('x').prepend_to_label('Prior Predictive: ')

    def posterior_predictive(self) -> StudentsT:
        return StudentsT(
            nu=2.0 * self.alpha_prime,
            mu=self._mu,
            sigma=sqrt(self.beta_prime / self.alpha_prime)
        ).with_x_label('x').prepend_to_label('Posterior Predictive: ')

    def rvs(self, num_samples: int = 1) -> np.ndarray:
        return self.posterior_predictive().rvs(num_samples)

    def plot(self, **kwargs) -> Figure:
        """
        Plot a 2x2 grid of prior, posterior, prior predictive, and posterior
        predictive distributions.
        """
        fig, axes = new_subplots(n_rows=2, n_cols=2)
        (ax_prior, ax_posterior,
         ax_prior_pred, ax_posterior_pred) = axes.flat

        self.prior().plot(ax=ax_prior, **kwargs)
        ax_prior.set_title('Prior: τ ~ Gamma')

        self.posterior().plot(ax=ax_posterior, **kwargs)
        ax_posterior.set_title('Posterior: τ | x ~ Gamma')

        self.prior_predictive().plot(ax=ax_prior_pred, **kwargs)
        ax_prior_pred.set_title("Prior Predictive: x' ~ Student's t")

        self.posterior_predictive().plot(ax=ax_posterior_pred, **kwargs)
        ax_posterior_pred.set_title("Posterior Predictive: x' | x ~ Student's t")

        return fig

    @staticmethod
    def infer_posterior(
            data: Series,
            mu: float,
            alpha: float = VaguePrior.Gamma.alpha,
            beta: float = VaguePrior.Gamma.beta,
            **kwargs
    ) -> Gamma:
        """
        Return the posterior Gamma distribution over precision τ given observed data.
        """
        n = len(data)
        sum_sq = float(power(data.values - mu, 2).sum())
        return Gamma(
            alpha=alpha + n / 2.0,
            beta=beta + sum_sq / 2.0
        ).with_x_label('τ').prepend_to_label('Posterior: ')

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            mu: float,
            alpha: float = VaguePrior.Gamma.alpha,
            beta: float = VaguePrior.Gamma.beta,
            **kwargs
    ) -> DataFrame:
        """
        Return a DataFrame mapping conditional variables to posterior Gamma
        distributions over precision τ.
        """
        if isinstance(prob_vars, str):
            prob_vars = [prob_vars]
        if isinstance(cond_vars, str):
            cond_vars = [cond_vars]
        cond_values = [
            list(data[cond_var].unique())
            for cond_var in cond_vars
        ]
        rows = []
        for cond_product in product(*cond_values):
            filter_dict = {
                cond_vars[i]: cond_product[i]
                for i in range(len(cond_vars))
            }
            filtered_data = data
            for k, v in filter_dict.items():
                filtered_data = filtered_data.loc[filtered_data[k] == v]
            for prob_var in prob_vars:
                row = {cond_vars[i]: cond_product[i]
                       for i in range(len(cond_vars))}
                row['prob_var'] = prob_var
                row['Gamma'] = GammaNormalConjugate.infer_posterior(
                    data=filtered_data[prob_var],
                    mu=mu,
                    alpha=alpha,
                    beta=beta
                )
                rows.append(row)
        return DataFrame(rows)

    def __str__(self):
        return (
            f'GammaNormalConjugate('
            f'α={num_format(self._alpha, 3)}, '
            f'β={num_format(self._beta, 3)}, '
            f'μ={num_format(self._mu, 3)}, '
            f'n={self._n})'
        )

    def __repr__(self):
        return (
            f'GammaNormalConjugate('
            f'alpha={self._alpha}, '
            f'beta={self._beta}, '
            f'mu={self._mu}, '
            f'n={self._n})'
        )


_GammaNormalConjugate = GammaNormalConjugate
