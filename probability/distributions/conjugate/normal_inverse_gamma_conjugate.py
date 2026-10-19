from math import sqrt
from typing import Optional, Union, Tuple

from matplotlib.figure import Figure
import numpy as np
from numpy import array, ndarray, power
from pandas import Series
from scipy.stats import invgamma, norm

from probability.custom_types.external_custom_types import Array1d
from probability.distributions.continuous.inverse_gamma import InverseGamma
from probability.distributions.continuous.students_t import StudentsT
from probability.distributions.mixins.conjugate import ConjugateMixin, \
    PredictiveMixin
from probability.plotting import new_subplots
from probability.utils import num_format


class NormalInverseGammaConjugate(
    ConjugateMixin,
    PredictiveMixin,
    object
):
    """
    Bayesian Normal model with both unknown mean μ and unknown variance σ²
    using a conjugate Normal-Inverse-Gamma prior.

    Prior:
        σ² ~ InverseGamma(α₀, β₀)
        μ | σ² ~ Normal(μ₀, σ² / κ₀)
        where κ₀ represents pseudo-observations for the mean.

    Likelihood:
        x_i ~ Normal(μ, σ²) for i = 1, ..., n

    Posterior:
        κ_n = κ₀ + n
        μ_n = (κ₀ μ₀ + n x̄) / κ_n
        α_n = α₀ + n / 2
        β_n = β₀ + 1/2 * sum((x_i - x̄)²) + (κ₀ * n * (x̄ - μ₀)²) / (2 * κ_n)

        Marginal posterior for μ:
            μ | x ~ Student's t(ν = 2α_n, loc = μ_n, scale = sqrt(β_n / (α_n * κ_n)))

        Marginal posterior for σ²:
            σ² | x ~ InverseGamma(α_n, β_n)

    Predictive:
        x_new | x ~ Student's t(ν = 2α_n, loc = μ_n, scale = sqrt(β_n * (κ_n + 1) / (α_n * κ_n)))
    """

    def __init__(
            self,
            mu_0: float = 0.0,
            kappa_0: float = 0.001,
            alpha_0: float = 0.001,
            beta_0: float = 0.001,
            x: Optional[Array1d] = None,
            n: Optional[int] = None,
            x_bar: Optional[float] = None,
            sum_sq_diff: Optional[float] = None
    ):
        """
        Create a new Normal-Inverse-Gamma conjugate distribution.

        :param mu_0: Prior location for the mean μ.
        :param kappa_0: Prior pseudo-observations count for the mean.
        :param alpha_0: Shape parameter of the Inverse-Gamma prior on σ².
        :param beta_0: Scale parameter of the Inverse-Gamma prior on σ².
        :param x: Observed data values.
        :param n: Number of observations (if x not directly passed).
        :param x_bar: Sample mean of observations.
        :param sum_sq_diff: Sum of squared differences from the sample mean sum((x_i - x̄)²).
        """
        self._mu_0: float = float(mu_0)
        self._kappa_0: float = float(kappa_0)
        self._alpha_0: float = float(alpha_0)
        self._beta_0: float = float(beta_0)

        if x is not None:
            self._x: Optional[ndarray] = array(x, dtype=float)
            self._n: int = len(self._x)
            if self._n > 0:
                self._x_bar: float = float(self._x.mean())
                self._sum_sq_diff: float = float(power(self._x - self._x_bar, 2).sum())
            else:
                self._x_bar = 0.0
                self._sum_sq_diff = 0.0
        elif n is not None:
            self._x = None
            self._n = int(n)
            self._x_bar = float(x_bar) if x_bar is not None else 0.0
            self._sum_sq_diff = float(sum_sq_diff) if sum_sq_diff is not None else 0.0
        else:
            self._x = None
            self._n = 0
            self._x_bar = 0.0
            self._sum_sq_diff = 0.0

    # region Hyperparameters

    @property
    def mu_0(self) -> float:
        return self._mu_0

    @mu_0.setter
    def mu_0(self, value: float):
        self._mu_0 = float(value)

    @property
    def kappa_0(self) -> float:
        return self._kappa_0

    @kappa_0.setter
    def kappa_0(self, value: float):
        self._kappa_0 = float(value)

    @property
    def alpha_0(self) -> float:
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, value: float):
        self._alpha_0 = float(value)

    @property
    def beta_0(self) -> float:
        return self._beta_0

    @beta_0.setter
    def beta_0(self, value: float):
        self._beta_0 = float(value)

    @property
    def x(self) -> Optional[ndarray]:
        return self._x

    @x.setter
    def x(self, value: Array1d):
        self._x = array(value, dtype=float)
        self._n = len(self._x)
        if self._n > 0:
            self._x_bar = float(self._x.mean())
            self._sum_sq_diff = float(power(self._x - self._x_bar, 2).sum())
        else:
            self._x_bar = 0.0
            self._sum_sq_diff = 0.0

    @property
    def n(self) -> int:
        return self._n

    @property
    def x_bar(self) -> float:
        return self._x_bar

    @property
    def sum_sq_diff(self) -> float:
        return self._sum_sq_diff

    # endregion

    # region Posterior Hyperparameters

    @property
    def kappa_n(self) -> float:
        return self._kappa_0 + self._n

    @property
    def mu_n(self) -> float:
        if self._n == 0:
            return self._mu_0
        return (self._kappa_0 * self._mu_0 + self._n * self._x_bar) / self.kappa_n

    @property
    def alpha_n(self) -> float:
        return self._alpha_0 + self._n / 2.0

    @property
    def beta_n(self) -> float:
        if self._n == 0:
            return self._beta_0
        mean_diff_sq = (self._x_bar - self._mu_0) ** 2
        mean_penalty = (self._kappa_0 * self._n * mean_diff_sq) / (2.0 * self.kappa_n)
        return self._beta_0 + self._sum_sq_diff / 2.0 + mean_penalty

    # endregion

    # region Marginal Distributions

    def prior_mu(self) -> StudentsT:
        """
        Return the marginal prior Student's t distribution for μ.
        """
        scale = sqrt(self._beta_0 / (self._alpha_0 * self._kappa_0))
        return StudentsT(
            nu=2.0 * self._alpha_0,
            mu=self._mu_0,
            sigma=scale
        ).with_x_label('μ').prepend_to_label('Prior: ')

    def posterior_mu(self) -> StudentsT:
        """
        Return the marginal posterior Student's t distribution for μ.
        """
        scale = sqrt(self.beta_n / (self.alpha_n * self.kappa_n))
        return StudentsT(
            nu=2.0 * self.alpha_n,
            mu=self.mu_n,
            sigma=scale
        ).with_x_label('μ').prepend_to_label('Posterior: ')

    def prior_sigma_sq(self) -> InverseGamma:
        """
        Return the marginal prior Inverse-Gamma distribution for σ².
        """
        return InverseGamma(
            alpha=self._alpha_0,
            beta=self._beta_0
        ).with_x_label('σ²').prepend_to_label('Prior: ')

    def posterior_sigma_sq(self) -> InverseGamma:
        """
        Return the marginal posterior Inverse-Gamma distribution for σ².
        """
        return InverseGamma(
            alpha=self.alpha_n,
            beta=self.beta_n
        ).with_x_label('σ²').prepend_to_label('Posterior: ')

    def prior(self, parameter: str = 'mu') -> Union[StudentsT, InverseGamma]:
        """
        Return the marginal prior distribution for the given parameter ('mu' or 'sigma_sq').
        """
        if parameter == 'mu':
            return self.prior_mu()
        elif parameter in ('sigma', 'sigma_sq', 'variance'):
            return self.prior_sigma_sq()
        else:
            raise ValueError(f"parameter must be 'mu' or 'sigma_sq', got {parameter}")

    def posterior(self, parameter: str = 'mu') -> Union[StudentsT, InverseGamma]:
        """
        Return the marginal posterior distribution for the given parameter ('mu' or 'sigma_sq').
        """
        if parameter == 'mu':
            return self.posterior_mu()
        elif parameter in ('sigma', 'sigma_sq', 'variance'):
            return self.posterior_sigma_sq()
        else:
            raise ValueError(f"parameter must be 'mu' or 'sigma_sq', got {parameter}")

    def prior_predictive(self) -> StudentsT:
        """
        Return the prior predictive distribution for a new observation x_new.
        """
        scale = sqrt(self._beta_0 * (self._kappa_0 + 1.0) / (self._alpha_0 * self._kappa_0))
        return StudentsT(
            nu=2.0 * self._alpha_0,
            mu=self._mu_0,
            sigma=scale
        ).with_x_label('x').prepend_to_label('Prior Predictive: ')

    def posterior_predictive(self) -> StudentsT:
        """
        Return the posterior predictive distribution for a new observation x_new | x.
        """
        scale = sqrt(self.beta_n * (self.kappa_n + 1.0) / (self.alpha_n * self.kappa_n))
        return StudentsT(
            nu=2.0 * self.alpha_n,
            mu=self.mu_n,
            sigma=scale
        ).with_x_label('x').prepend_to_label('Posterior Predictive: ')

    def rvs(self, num_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the posterior predictive distribution.
        """
        return self.posterior_predictive().rvs(num_samples)

    def sample_posterior(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw joint (μ, σ²) samples from the posterior distribution.

        :return: (mu_samples, sigma_sq_samples)
        """
        sigma_sq_samples = invgamma.rvs(
            self.alpha_n, scale=self.beta_n, size=num_samples
        )
        mu_scales = np.sqrt(sigma_sq_samples / self.kappa_n)
        mu_samples = norm.rvs(loc=self.mu_n, scale=mu_scales)
        return mu_samples, sigma_sq_samples

    # endregion

    def plot(self, **kwargs) -> Figure:
        """
        Plot a 2x3 grid showing prior and posterior for μ, σ², and predictive distributions.
        """
        fig, axes = new_subplots(n_rows=2, n_cols=3)
        (ax_mu_prior, ax_mu_post, ax_pred,
         ax_sig_prior, ax_sig_post, ax_joint) = axes.flat

        # Row 1: Mean and Predictive
        self.prior_mu().plot(ax=ax_mu_prior, **kwargs)
        ax_mu_prior.set_title("Prior: μ ~ Student's t")

        self.posterior_mu().plot(ax=ax_mu_post, **kwargs)
        ax_mu_post.set_title("Posterior: μ | x ~ Student's t")

        self.prior_predictive().plot(ax=ax_pred, color='gray', linestyle='--', label='Prior Pred')
        self.posterior_predictive().plot(ax=ax_pred, color='blue', label='Posterior Pred')
        ax_pred.legend()
        ax_pred.set_title("Predictive: x' ~ Student's t")

        # Row 2: Variance and Joint
        self.prior_sigma_sq().plot(ax=ax_sig_prior, **kwargs)
        ax_sig_prior.set_title("Prior: σ² ~ InvGamma")

        self.posterior_sigma_sq().plot(ax=ax_sig_post, **kwargs)
        ax_sig_post.set_title("Posterior: σ² | x ~ InvGamma")

        # Joint posterior scatter of samples
        mu_s, sig_s = self.sample_posterior(1000)
        ax_joint.scatter(mu_s, sig_s, alpha=0.3, s=10, color='purple')
        ax_joint.set_xlabel('μ')
        ax_joint.set_ylabel('σ²')
        ax_joint.set_title('Joint Posterior (μ, σ²) Samples')

        return fig

    @classmethod
    def infer_posterior(
            cls,
            data: Union[Series, Array1d],
            mu_0: float = 0.0,
            kappa_0: float = 0.001,
            alpha_0: float = 0.001,
            beta_0: float = 0.001
    ) -> 'NormalInverseGammaConjugate':
        """
        Fit a NormalInverseGammaConjugate model directly to continuous data.
        """
        values = data.values if isinstance(data, Series) else array(data)
        return cls(
            mu_0=mu_0,
            kappa_0=kappa_0,
            alpha_0=alpha_0,
            beta_0=beta_0,
            x=values
        )

    def __str__(self):
        return (
            f'NormalInverseGammaConjugate('
            f'μ₀={num_format(self._mu_0, 3)}, '
            f'κ₀={num_format(self._kappa_0, 3)}, '
            f'α₀={num_format(self._alpha_0, 3)}, '
            f'β₀={num_format(self._beta_0, 3)}, '
            f'n={self._n})'
        )

    def __repr__(self):
        return (
            f'NormalInverseGammaConjugate('
            f'mu_0={self._mu_0}, '
            f'kappa_0={self._kappa_0}, '
            f'alpha_0={self._alpha_0}, '
            f'beta_0={self._beta_0}, '
            f'n={self._n})'
        )
