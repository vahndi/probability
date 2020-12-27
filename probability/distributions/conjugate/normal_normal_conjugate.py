from itertools import product
from typing import overload, Optional, Union, List

from matplotlib.figure import Figure
from mpl_format.figures import FigureFormatter
from numpy.ma import arange
from pandas import DataFrame, Series

from probability.distributions.continuous.normal import Normal
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin

from probability.utils import none_are_none, all_are_none, num_format, \
    one_is_none


class NormalNormalConjugate(
    ConjugateMixin,
    PredictiveMixin,
    object
):
    """
    Class for estimating the mean of a normal distribution with
    known variance σ² or known precision τ (where τ = 1 / σ²).

    Prior Hyper-parameters
    ----------------------
    * `μ₀` and one of `σ₀²` or `τ₀` are the hyper-parameters for the value of
    the normally-distributed mean of the distribution.
    * `μ₀` is the mean value of the mean.
    * `σ₀²` is the variance of the mean.
    * `τ₀` is the precision (inverse of the variance) of the mean.
    * Interpretation is that the mean was estimated from prior observations with
        - total precision (sum of all individual precisions) τ₀ = 1 / σ₀²
        - sample mean μ₀.

    Posterior Hyper-parameters
    --------------------------
    * `n` is the number of observations.
    * `x_sum` is the sum of the observations Σx.
    * `σ²` is the variance of the observations.
    * `τ` is the precision (inverse of the variance) of the observations

    Model parameters
    ----------------
    * `P(x)` is the probability that μ = x.
    * -∞ ≤ x ≤ ∞

    Links
    -----
    * https://en.wikipedia.org/wiki/Normal_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior
    """
    @overload
    def __init__(
            self, n: int,
            x_sum: float,
            sigma_sq: float,
            mu_0: float,
            sigma_0_sq: float
    ):
        """
        Create a new normal-normal conjugate distribution.

        :param n: Number of observations.
        :param x_sum: Sum of observations.
        :param sigma_sq: Variance of observations.
        :param mu_0: Value for the μ₀ (mean) hyper-parameter of the prior Normal
                     distribution describing the mean.
        :param sigma_0_sq: Value for the σ₀² (variance) hyper-parameter of the
                           prior Normal distribution describing the mean.
        """
        pass

    @overload
    def __init__(
            self, n: int,
            x_sum: float,
            tau: float,
            mu_0: float,
            tau_0: float,
    ):
        """
        Create a new normal-normal conjugate distribution.

        :param n: Number of observations.
        :param x_sum: Sum of observations.
        :param tau: Variance of observations.
        :param mu_0: Value for the μ₀ (mean) hyper-parameter of the prior Normal
                     distribution describing the mean.
        :param tau_0: Value for the τ₀ (precision) hyper-parameter of the
                      prior Normal distribution describing the mean.
        """
        pass

    def __init__(
            self, n: int,
            x_sum: float,
            mu_0: float,
            sigma_sq: Optional[float] = None,
            tau: Optional[float] = None,
            sigma_0_sq: Optional[float] = None,
            tau_0: Optional[float] = None
    ):
        """
        Create a new normal-normal conjugate distribution.

        :param n: Number of observations.
        :param x_sum: Sum of observations.
        :param sigma_sq: Variance of observations.
                         Distribution describing the mean.
        :param tau: Precision of observations.
        :param mu_0: Value for the μ₀ (mean) hyper-parameter of the prior Normal
        :param sigma_0_sq: Value for the σ₀² (variance) hyper-parameter of the
                           prior Normal distribution describing the mean.
        :param tau_0: Value for the τ₀ (precision) hyper-parameter of the
                      prior Normal distribution describing the mean.
        """
        if not (
            none_are_none(sigma_0_sq, sigma_sq) and
            all_are_none(tau_0, tau)
        ) and not (
            all_are_none(sigma_0_sq, sigma_sq) and
            none_are_none(tau_0, tau)
        ):
            raise ValueError('Give either σ² and σ₀² or τ and τ₀')

        self._n: int = n
        self._x_sum: float = x_sum
        self._mu_0: float = mu_0
        if sigma_0_sq is not None:
            self._sigma_sq: float = sigma_sq
            self._sigma_0_sq: float = sigma_0_sq
            self._parameterization: str = 'σ²'
        else:
            self._sigma_sq: float = 1 / tau
            self._sigma_0_sq: float = 1 / tau_0
            self._parameterization: str = 'τ'

    @property
    def mu_0(self) -> float:
        return self._mu_0

    @mu_0.setter
    def mu_0(self, value: float):
        self._mu_0 = value

    @property
    def sigma_sq(self) -> float:
        return self._sigma_sq

    @sigma_sq.setter
    def sigma_sq(self, value: float):
        self._sigma_sq = value
        self._parameterization = 'σ²'

    @property
    def sigma_0_sq(self) -> float:
        return self._sigma_0_sq

    @sigma_0_sq.setter
    def sigma_0_sq(self, value: float):
        self._sigma_0_sq = value
        self._parameterization = 'σ²'

    @property
    def tau(self) -> float:
        return 1 / self._sigma_sq

    @tau.setter
    def tau(self, value: float):
        self._sigma_sq = 1 / value
        self._parameterization = 'τ'

    @property
    def tau_0(self) -> float:
        return 1 / self._sigma_0_sq

    @tau_0.setter
    def tau_0(self, value: float):
        self._sigma_0_sq = 1 / value
        self._parameterization = 'τ'

    @property
    def n(self) -> int:
        return self._n

    # region posterior hyper-parameters

    @property
    def mu_0_prime(self) -> float:
        return (
            (self.tau_0 * self.mu_0 + self.tau * self._x_sum) /
            (self.tau_0 + self.n * self.tau)
        )

    @property
    def tau_0_prime(self) -> float:
        return self.tau_0 + self.n * self.tau

    @property
    def sigma_0_sq_prime(self) -> float:
        return 1 / self.tau_0_prime

    # endregion

    def prior(self) -> Normal:
        """
        Return a Normal distribution reflecting the prior belief about the
        distribution of the parameter μ, before seeing any data.
        """
        if self._parameterization == 'σ²':
            y_label = (
                '$P(μ_{Norm}=x|'
                'μ_{0, Norm},'
                'σ²_{0, Norm})$'
            )
        else:
            y_label = (
                '$P(μ_{Norm}=x|'
                'μ_{0, Norm},'
                'τ_{0, Norm})$'
            )
        return Normal(
            mu=self.mu_0,
            sigma_sq=self.sigma_0_sq
        ).with_x_label('μ').with_y_label(y_label).prepend_to_label('Prior: ')

    def likelihood(self, **kwargs) -> Normal:
        """
        Return a distribution reflecting the likelihood of observing
        the data, under a Normal model, independent of the prior belief
        about the distribution of parameter μ.
        """
        return Normal(
            mu=self._x_sum / self._n,
            sigma_sq=self._sigma_sq
        )

    def posterior(self) -> Normal:
        """
        Return a Normal distribution reflecting the posterior belief about the
        distribution of the parameter μ, after observing the data.
        """
        if self._parameterization == 'σ²':
            y_label = (
                '$P(μ_{Norm}=x|'
                'μ_{0, Norm}′,'
                'σ²_{0, Norm}′)$'
            )
        else:
            y_label = (
                '$P(μ_{Norm}=x|'
                'μ_{0, Norm}′,'
                'τ_{0, Norm}′)$'
            )

        return Normal(
            mu=self.mu_0_prime,
            sigma_sq=self.sigma_0_sq_prime
        ).with_x_label('μ').with_y_label(y_label).prepend_to_label(
            'Posterior: '
        )

    # region predictive

    def prior_predictive(self) -> Normal:
        """
        Return a Normal describing the expected distribution of future
        values based on the prior parameter estimates.
        """
        return Normal(
            mu=self.mu_0,
            sigma_sq=self.sigma_0_sq + self.sigma_sq
        ).with_y_label(
            r'$P(\tilde{X}=x|'
            r'μ_{Norm}=μ_{0, Norm},'
            r'σ²=σ²_{0, Norm}+σ²)$'
        )

    def posterior_predictive(self) -> Normal:
        """
        Return a Normal describing the expected distribution of future
        values based on the posterior parameter estimates.
        """
        return Normal(
            mu=self.mu_0_prime,
            sigma_sq=self.sigma_0_sq_prime + self.sigma_sq
        ).with_y_label(
            r'$P(\tilde{X}=x|'
            r'μ_{Norm}=μ_{0, Norm}′,'
            r'σ²=σ²_{0, Norm}′+σ²)$'
        )

    # endregion

    def plot(self, **kwargs) -> Figure:
        """
        Plot a grid of the different components of the Compound Distribution.

        :param kwargs: kwargs for plot methods
        """
        ppf_prior_01 = self.prior().ppf().at(0.01)
        ppf_prior_99 = self.prior().ppf().at(0.99)
        ppf_posterior_01 = self.posterior().ppf().at(0.99)
        ppf_posterior_99 = self.posterior().ppf().at(0.99)
        x_norm_min = int(min(ppf_prior_01, ppf_posterior_01)) - 1
        x_norm_max = int(max(ppf_prior_99, ppf_posterior_99)) + 1
        x_param = arange(x_norm_min, x_norm_max + 0.001, 0.001)
        ff = FigureFormatter(n_rows=2, n_cols=3)
        (
            ax_prior, ax_data, ax_posterior,
            ax_prior_predictive, ax_likelihood, ax_posterior_predictive
        ) = ff.axes.flat
        # plot prior and posterior parameters
        self.prior().plot(x=x_param, ax=ax_prior.axes, **kwargs)
        self.posterior().plot(x=x_param, ax=ax_posterior.axes, **kwargs)
        y_max_params = max(ax_prior.get_y_max(), ax_posterior.get_y_max())
        ax_prior.set_y_lim(0, y_max_params)
        ax_posterior.set_y_lim(0, y_max_params)
        ax_prior.set_title_text('prior').add_legend()
        ax_posterior.set_title_text('posterior').add_legend()
        # plot prior and posterior predictives
        ppf_prior_pred_01 = self.prior_predictive().ppf().at(0.01)
        ppf_prior_pred_99 = self.prior_predictive().ppf().at(0.99)
        ppf_posterior_pred_01 = self.posterior_predictive().ppf().at(0.01)
        ppf_posterior_pred_99 = self.posterior_predictive().ppf().at(0.99)
        x_pred_min = int(min(ppf_prior_pred_01, ppf_posterior_pred_01)) - 1
        x_pred_max = int(max(ppf_prior_pred_99, ppf_posterior_pred_99)) + 1
        x_pred = arange(x_pred_min, x_pred_max + 0.001, 0.001)
        self.prior_predictive().plot(
            x=x_pred, kind='line',
            ax=ax_prior_predictive.axes,
            **kwargs
        )
        self.posterior_predictive().plot(
            x=x_pred, kind='line',
            ax=ax_posterior_predictive.axes,
            **kwargs
        )
        y_max_pred = max(ax_prior_predictive.get_y_max(),
                         ax_posterior_predictive.get_y_max())
        ax_prior_predictive.set_y_lim(0, y_max_pred)
        ax_posterior_predictive.set_y_lim(0, y_max_pred)
        ax_prior_predictive.set_title_text('prior predictive').add_legend()
        ax_posterior_predictive.set_title_text(
            'posterior predictive'
        ).add_legend()
        # plot data
        observations = self.likelihood().rvs(num_samples=self._n)
        observations.plot.bar(ax=ax_data.axes, **kwargs)
        ax_data.set_text(title='data', x_label='i', y_label='$X_i$')
        y_abs_max = max([abs(y_lim) for y_lim in ax_data.get_y_lim()])
        ax_data.set_y_lim(-y_abs_max, y_abs_max)
        # plot likelihood
        x_like_min = int(self.likelihood().ppf().at(0.01)) - 1
        x_like_max = int(self.likelihood().ppf().at(0.99)) + 1
        x_exponential = arange(x_like_min, x_like_max + 0.001, 0.001)
        self.likelihood().plot(x=x_exponential, ax=ax_likelihood.axes)
        ax_likelihood.set_title_text('likelihood')
        ax_likelihood.add_legend()
        return ff.figure

    @staticmethod
    @overload
    def infer_posterior(
            data: Series,
            mu_0: float,
            sigma_0_sq: Optional[float] = None,
    ) -> Normal:
        pass

    @staticmethod
    @overload
    def infer_posterior(
            data: Series,
            mu_0: float,
            tau_0: Optional[float] = None
    ) -> Normal:
        pass

    @staticmethod
    def infer_posterior(
            data: Series,
            mu_0: float,
            sigma_0_sq: Optional[float] = None,
            tau_0: Optional[float] = None
    ) -> Normal:
        """
        Return a new Normal distribution of the posterior most likely to
        generate the given data.

        :param data: Series of float observations.
        :param mu_0: Value for the μ₀ (mean) hyper-parameter of the prior Normal
                     distribution describing the mean.
        :param sigma_0_sq: Value for the σ₀² (variance) hyper-parameter of the
                           prior Normal distribution describing the mean.
        :param tau_0: Value for the τ₀ (precision) hyper-parameter of the
                      prior Normal distribution describing the mean.
        """
        if not one_is_none(sigma_0_sq, tau_0):
            raise ValueError('Give either σ₀² or τ₀')

        n = len(data)
        x_sum = data.sum()
        if sigma_0_sq is None:
            tau = 1 / data.var()
            return NormalNormalConjugate(
                n=n, x_sum=x_sum, mu_0=mu_0,
                tau=tau, tau_0=tau_0
            ).posterior()
        else:
            sigma_sq = data.var()
            return NormalNormalConjugate(
                n=n, x_sum=x_sum, mu_0=mu_0,
                sigma_sq=sigma_sq, sigma_0_sq=sigma_0_sq
            ).posterior()

    @staticmethod
    @overload
    def infer_posteriors(
            data: DataFrame, prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            mu_0: float,
            sigma_0_sq: Optional[float] = None,
            stats: Optional[Union[str, List[str]]] = None
    ) -> DataFrame:

        pass

    @staticmethod
    @overload
    def infer_posteriors(
            data: DataFrame, prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            mu_0: float,
            tau_0: Optional[float] = None,
            stats: Optional[Union[str, List[str]]] = None
    ) -> DataFrame:

        pass

    @staticmethod
    def infer_posteriors(
            data: DataFrame, prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            mu_0: float,
            sigma_0_sq: Optional[float] = None,
            tau_0: Optional[float] = None,
            stats: Optional[Union[str, List[str]]] = None,
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
        :param mu_0: Value for the μ₀ (mean) hyper-parameter of the prior Normal
                     distribution describing the mean.
        :param sigma_0_sq: Value for the σ₀² (variance) hyper-parameter of the
                           prior Normal distribution describing the mean.
        :param tau_0: Value for the τ₀ (precision) hyper-parameter of the
                      prior Normal distribution describing the mean.
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
        mus = []
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
                if sigma_0_sq is not None:
                    posterior = NormalNormalConjugate.infer_posterior(
                        data=cond_data[prob_var],
                        mu_0=mu_0, sigma_0_sq=sigma_0_sq
                    )
                else:
                    posterior = NormalNormalConjugate.infer_posterior(
                        data=cond_data[prob_var],
                        mu_0=mu_0, tau_0=tau_0
                    )
                prob_dict['mu'] = posterior
                for stat in stats:
                    prob_dict = {**prob_dict,
                                 **posterior.stat(stat, True)}
                mus.append(prob_dict)

        mus_data = DataFrame(mus)

        return mus_data

    def __str__(self):

        if self._parameterization == 'σ²':
            return (
                f'NormalNormalConjugate('
                f'n={self._n}, '
                f'Σx={self._x_sum}, '
                f'σ²={num_format(self._sigma_sq, 3)})'
                f'μ₀={num_format(self._mu_0, 3)}, '
                f'σ₀²={num_format(self._sigma_0_sq, 3)}, '
            )
        elif self._parameterization == 'τ':
            return (
                f'NormalNormalConjugate('
                f'n={self._n}, '
                f'Σx={self._x_sum}, '
                f'τ={num_format(self.tau, 3)})'
                f'μ₀={num_format(self._mu_0, 3)}, '
                f'τ₀={num_format(self.tau_0, 3)}, '
            )
        else:
            return 'Invalid parameterization!'

    def __repr__(self):

        if self._parameterization == 'σ²':
            return (
                f'NormalNormalConjugate('
                f'n={self._n}, '
                f'x_sum={self._x_sum}, '
                f'sigma_sq={self._sigma_sq})'
                f'mu_0={self._mu_0}, '
                f'sigma_0_sq={self._sigma_0_sq}, '
            )
        elif self._parameterization == 'τ':
            return (
                f'NormalNormalConjugate('
                f'n={self._n}, '
                f'x_sum={self._x_sum}, '
                f'tau={self.tau})'
                f'mu_0={self._mu_0}, '
                f'tau_0={self.tau_0}, '
            )
        else:
            return 'Invalid parameterization!'
