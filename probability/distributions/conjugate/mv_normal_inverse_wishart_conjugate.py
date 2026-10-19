from math import sqrt
from typing import Optional, Union, Tuple, List

from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
import numpy as np
from numpy import ndarray
from pandas import Series, DataFrame
from scipy.stats import invwishart

from probability.custom_types.external_custom_types import FloatArray1d, FloatArray2d
from probability.distributions.continuous.inverse_gamma import InverseGamma
from probability.distributions.continuous.students_t import StudentsT
from probability.distributions.multivariate.inverse_wishart import InverseWishart
from probability.distributions.multivariate.mv_students_t import MVStudentsT
from probability.distributions.mixins.conjugate import ConjugateMixin, PredictiveMixin
from probability.plotting import new_axes


class MVNormalInverseWishartConjugate(
    ConjugateMixin,
    PredictiveMixin,
    object
):
    """
    Multivariate Normal model with unknown mean vector μ and unknown covariance matrix Σ
    using a conjugate Normal-Inverse-Wishart prior.

    Prior:
        Σ ~ InverseWishart(ν₀, Ψ₀)
        μ | Σ ~ MVNormal(μ₀, Σ / κ₀)
        where κ₀ represents pseudo-observations for the mean,
        and ν₀ > d - 1 represents degrees of freedom for the covariance.

    Likelihood:
        x_i ~ MVNormal(μ, Σ) for i = 1, ..., n

    Posterior:
        κ_n = κ₀ + n
        ν_n = ν₀ + n
        μ_n = (κ₀ μ₀ + n x̄) / κ_n
        Ψ_n = Ψ₀ + S + (κ₀ * n / κ_n) (x̄ - μ₀)(x̄ - μ₀)ᵀ
        where S = sum_{i=1}^n (x_i - x̄)(x_i - x̄)ᵀ is the sample scatter matrix.

    Marginal Posteriors:
        Σ | X ~ InverseWishart(ν_n, Ψ_n)
        μ | X ~ MVStudentsT(ν = ν_n - d + 1, loc = μ_n, shape = Ψ_n / (κ_n * (ν_n - d + 1)))

    Predictive Distribution:
        x_new | X ~ MVStudentsT(ν = ν_n - d + 1, loc = μ_n, shape = Ψ_n * (κ_n + 1) / (κ_n * (ν_n - d + 1)))
    """

    def __init__(
            self,
            mu_0: Optional[Union[FloatArray1d, dict, Series]] = None,
            kappa_0: float = 0.001,
            nu_0: Optional[float] = None,
            psi_0: Optional[Union[FloatArray2d, DataFrame]] = None,
            x: Optional[Union[DataFrame, ndarray]] = None,
            n: Optional[int] = None,
            x_bar: Optional[Union[Series, ndarray]] = None,
            scatter: Optional[Union[DataFrame, ndarray]] = None,
            names: Optional[List[str]] = None
    ):
        """
        Create a new Normal-Inverse-Wishart conjugate model.
        """
        # Determine names and dimension
        if x is not None:
            if isinstance(x, DataFrame):
                names = list(x.columns) if names is None else names
                d = x.shape[1]
                data_arr = x.to_numpy(dtype=float)
            else:
                data_arr = np.array(x, dtype=float)
                d = data_arr.shape[1]
                if names is None:
                    names = [f'x{k}' for k in range(1, d + 1)]
            n = data_arr.shape[0]
            x_bar_arr = data_arr.mean(axis=0)
            diff_data = data_arr - x_bar_arr
            scatter_arr = diff_data.T @ diff_data
        else:
            if psi_0 is not None:
                d = psi_0.shape[0] if hasattr(psi_0, 'shape') else len(psi_0)
                if names is None and isinstance(psi_0, DataFrame):
                    names = list(psi_0.columns)
            elif mu_0 is not None:
                d = len(mu_0)
                if names is None and isinstance(mu_0, (dict, Series)):
                    names = list(mu_0.keys())
            else:
                d = 2
            if names is None:
                names = [f'x{k}' for k in range(1, d + 1)]

            x_bar_arr = np.array(x_bar, dtype=float) if x_bar is not None else None
            scatter_arr = np.array(scatter, dtype=float) if scatter is not None else None
            n = n if n is not None else (0 if x_bar_arr is None else 1)

        self._dim = d
        self._names = names

        # Initialize prior parameters
        if mu_0 is None:
            mu_0_arr = np.zeros(d, dtype=float)
        elif isinstance(mu_0, (dict, Series)):
            mu_0_arr = np.array([mu_0[k] for k in names], dtype=float)
        else:
            mu_0_arr = np.array(mu_0, dtype=float)

        nu_0_val = float(nu_0) if nu_0 is not None else float(d + 2.0)
        if nu_0_val <= d - 1:
            raise ValueError(f"Prior degrees of freedom nu_0 ({nu_0_val}) must be greater than d - 1 = {d - 1}.")

        if psi_0 is None:
            psi_0_arr = np.eye(d, dtype=float)
        elif isinstance(psi_0, DataFrame):
            psi_0_arr = psi_0.to_numpy(dtype=float)
        else:
            psi_0_arr = np.array(psi_0, dtype=float)

        self._mu_0 = mu_0_arr
        self._kappa_0 = float(kappa_0)
        self._nu_0 = nu_0_val
        self._psi_0 = psi_0_arr

        self._n = n
        self._x_bar = x_bar_arr
        self._scatter = scatter_arr

        # Compute posterior updates
        if self._n > 0 and self._x_bar is not None and self._scatter is not None:
            self._kappa_n = self._kappa_0 + self._n
            self._nu_n = self._nu_0 + self._n
            self._mu_n = (self._kappa_0 * self._mu_0 + self._n * self._x_bar) / self._kappa_n
            diff_mean = self._x_bar - self._mu_0
            outer_diff = np.outer(diff_mean, diff_mean)
            self._psi_n = (
                self._psi_0 + self._scatter +
                (self._kappa_0 * self._n / self._kappa_n) * outer_diff
            )
        else:
            self._kappa_n = self._kappa_0
            self._nu_n = self._nu_0
            self._mu_n = self._mu_0.copy()
            self._psi_n = self._psi_0.copy()

    @property
    def dim(self) -> int:
        """Dimension d."""
        return self._dim

    @property
    def names(self) -> List[str]:
        """Variable names."""
        return self._names

    @property
    def n(self) -> int:
        """Sample count."""
        return self._n

    @property
    def kappa_0(self) -> float:
        """Prior kappa."""
        return self._kappa_0

    @property
    def nu_0(self) -> float:
        """Prior degrees of freedom."""
        return self._nu_0

    @property
    def mu_0(self) -> Series:
        """Prior mean vector."""
        return Series(self._mu_0, index=self._names)

    @property
    def psi_0(self) -> DataFrame:
        """Prior scale matrix."""
        return DataFrame(self._psi_0, index=self._names, columns=self._names)

    @property
    def kappa_n(self) -> float:
        """Posterior kappa."""
        return self._kappa_n

    @property
    def nu_n(self) -> float:
        """Posterior degrees of freedom."""
        return self._nu_n

    @property
    def mu_n(self) -> Series:
        """Posterior mean vector."""
        return Series(self._mu_n, index=self._names)

    @property
    def psi_n(self) -> DataFrame:
        """Posterior scale matrix."""
        return DataFrame(self._psi_n, index=self._names, columns=self._names)

    # -------------------------------------------------------------------------
    # Prior and Posterior Distributions
    # -------------------------------------------------------------------------

    def prior_sigma(self) -> InverseWishart:
        """
        Prior marginal distribution for covariance matrix Sigma:
            Sigma ~ InverseWishart(nu_0, Psi_0)
        """
        return InverseWishart(
            nu=self._nu_0,
            psi=self._psi_0,
            names=self._names
        )

    def prior_mu(self) -> MVStudentsT:
        """
        Prior marginal distribution for mean vector mu:
            mu ~ MVStudentsT(nu = nu_0 - d + 1, loc = mu_0, shape = Psi_0 / (kappa_0 * (nu_0 - d + 1)))
        """
        df = self._nu_0 - self._dim + 1.0
        scale = self._psi_0 / (self._kappa_0 * df)
        return MVStudentsT(
            nu=df,
            mu=self._mu_0,
            sigma=scale,
            names=self._names
        )

    def prior(self) -> MVStudentsT:
        """
        Alias for prior_mu() returning the prior distribution for mu.
        """
        return self.prior_mu()

    def posterior_sigma(self) -> InverseWishart:
        """
        Posterior marginal distribution for covariance matrix Sigma:
            Sigma | X ~ InverseWishart(nu_n, Psi_n)
        """
        return InverseWishart(
            nu=self._nu_n,
            psi=self._psi_n,
            names=self._names
        )

    def posterior_mu(self) -> MVStudentsT:
        """
        Posterior marginal distribution for mean vector mu:
            mu | X ~ MVStudentsT(nu = nu_n - d + 1, loc = mu_n, shape = Psi_n / (kappa_n * (nu_n - d + 1)))
        """
        df = self._nu_n - self._dim + 1.0
        scale = self._psi_n / (self._kappa_n * df)
        return MVStudentsT(
            nu=df,
            mu=self._mu_n,
            sigma=scale,
            names=self._names
        )

    def posterior(self) -> MVStudentsT:
        """
        Alias for posterior_mu() returning the posterior distribution for mu.
        """
        return self.posterior_mu()

    def prior_predictive(self) -> MVStudentsT:
        """
        Prior predictive distribution for a new observation x_new:
            x_new ~ MVStudentsT(nu = nu_0 - d + 1, loc = mu_0, shape = Psi_0 * (kappa_0 + 1) / (kappa_0 * (nu_0 - d + 1)))
        """
        df = self._nu_0 - self._dim + 1.0
        scale = self._psi_0 * (self._kappa_0 + 1.0) / (self._kappa_0 * df)
        return MVStudentsT(
            nu=df,
            mu=self._mu_0,
            sigma=scale,
            names=self._names
        )

    def posterior_predictive(self) -> MVStudentsT:
        """
        Posterior predictive distribution for a new observation x_new:
            x_new | X ~ MVStudentsT(nu = nu_n - d + 1, loc = mu_n, shape = Psi_n * (kappa_n + 1) / (kappa_n * (nu_n - d + 1)))
        """
        df = self._nu_n - self._dim + 1.0
        scale = self._psi_n * (self._kappa_n + 1.0) / (self._kappa_n * df)
        return MVStudentsT(
            nu=df,
            mu=self._mu_n,
            sigma=scale,
            names=self._names
        )

    # -------------------------------------------------------------------------
    # Marginal Univariate Posteriors
    # -------------------------------------------------------------------------

    def marginal_mu(self, name_or_idx: Union[str, int]) -> StudentsT:
        """
        Return the exact 1D marginal posterior Student's t distribution for a single component mu_j.
        """
        return self.posterior_mu().marginal(name_or_idx)

    def marginal_sigma_sq(self, name_or_idx: Union[str, int]) -> InverseGamma:
        """
        Return the exact 1D marginal posterior Inverse-Gamma distribution for a single variance Sigma_jj.
        """
        return self.posterior_sigma().marginal_variance(name_or_idx)

    # -------------------------------------------------------------------------
    # Posterior Sampling
    # -------------------------------------------------------------------------

    def sample_posterior(
            self,
            num_samples: int = 1000,
            random_state: Optional[int] = None
    ) -> Tuple[DataFrame, List[DataFrame]]:
        """
        Draw joint (mu, Sigma) samples from the exact posterior.

        :param num_samples: Number of posterior draws.
        :param random_state: Optional seed for reproducibility.
        :return: (mu_samples_df, sigma_samples_list)
        """
        if random_state is not None:
            np.random.seed(random_state)

        sigma_samples = invwishart(
            df=self._nu_n,
            scale=self._psi_n
        ).rvs(size=num_samples, random_state=random_state)

        if num_samples == 1 and sigma_samples.ndim == 2:
            sigma_samples = np.expand_dims(sigma_samples, axis=0)

        mu_samples = []
        sigma_dfs = []
        for s in range(num_samples):
            sig_s = sigma_samples[s]
            cov_mu = sig_s / self._kappa_n
            mu_s = np.random.multivariate_normal(mean=self._mu_n, cov=cov_mu)
            mu_samples.append(mu_s)
            sigma_dfs.append(DataFrame(sig_s, index=self._names, columns=self._names))

        mu_df = DataFrame(mu_samples, columns=self._names)
        mu_df.index.name = 'sample_index'
        return mu_df, sigma_dfs

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def infer_posterior(
            cls,
            data: Union[DataFrame, ndarray],
            kappa_0: float = 0.001,
            nu_0: Optional[float] = None
    ) -> 'MVNormalInverseWishartConjugate':
        """
        Fit a conjugate Normal-Inverse-Wishart model from continuous multivariate data.
        """
        return cls(
            x=data,
            kappa_0=kappa_0,
            nu_0=nu_0
        )

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def _draw_ellipse(
            self,
            ax: Axes,
            center: Tuple[float, float],
            cov_2d: ndarray,
            p_val: float = 0.95,
            edgecolor: str = 'black',
            linestyle: str = '-',
            label: Optional[str] = None,
            alpha: float = 0.3
    ):
        """Helper to draw a 2D credible ellipse."""
        vals, vecs = np.linalg.eigh(cov_2d)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # chi2 critical value for 2 df: -2 * ln(1 - p_val)
        s = sqrt(-2.0 * np.log(1.0 - p_val))
        width = 2.0 * s * sqrt(max(vals[0], 1e-12))
        height = 2.0 * s * sqrt(max(vals[1], 1e-12))

        ell = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=theta,
            edgecolor=edgecolor,
            facecolor='none',
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
            alpha=alpha
        )
        ax.add_patch(ell)

    def plot(
            self,
            dim1: Union[str, int] = 0,
            dim2: Union[str, int] = 1,
            p_level: float = 0.95,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot 2D projections of the observed data, prior vs. posterior mean,
        and 95% credible ellipses for the mean vector μ and predictive x_new.
        """
        ax = ax or new_axes()

        if isinstance(dim1, int):
            i1 = dim1
            name1 = self._names[i1]
        else:
            name1 = dim1
            i1 = self._names.index(name1)

        if isinstance(dim2, int):
            i2 = dim2
            name2 = self._names[i2]
        else:
            name2 = dim2
            i2 = self._names.index(name2)

        idx_2d = [i1, i2]

        # Prior center & 2D cov
        prior_center = (float(self._mu_0[i1]), float(self._mu_0[i2]))
        prior_cov_mu = (
            self._psi_0[np.ix_(idx_2d, idx_2d)] /
            (self._kappa_0 * (self._nu_0 - self._dim + 1.0))
        )
        self._draw_ellipse(
            ax=ax,
            center=prior_center,
            cov_2d=prior_cov_mu,
            p_val=p_level,
            edgecolor='tab:red',
            linestyle=':',
            label=f'Prior Mean {p_level:.0%} Ellipse'
        )
        ax.plot(prior_center[0], prior_center[1], marker='x', markersize=8,
                color='tab:red', label=r'Prior $\mu_0$')

        # Posterior center & 2D cov
        post_center = (float(self._mu_n[i1]), float(self._mu_n[i2]))
        df_post = self._nu_n - self._dim + 1.0
        post_cov_mu = (
            self._psi_n[np.ix_(idx_2d, idx_2d)] /
            (self._kappa_n * df_post)
        )
        self._draw_ellipse(
            ax=ax,
            center=post_center,
            cov_2d=post_cov_mu,
            p_val=p_level,
            edgecolor='tab:blue',
            linestyle='-',
            label=f'Posterior Mean {p_level:.0%} Ellipse'
        )
        ax.plot(post_center[0], post_center[1], marker='o', markersize=8,
                color='tab:blue', label=r'Posterior $\mu_n$')

        # Posterior Predictive ellipse
        pred_cov = (
            self._psi_n[np.ix_(idx_2d, idx_2d)] *
            (self._kappa_n + 1.0) / (self._kappa_n * df_post)
        )
        self._draw_ellipse(
            ax=ax,
            center=post_center,
            cov_2d=pred_cov,
            p_val=p_level,
            edgecolor='tab:green',
            linestyle='--',
            label=f'Predictive $\\tilde{{x}}$ {p_level:.0%} Ellipse'
        )

        # Plot sample mean if data was present
        if self._x_bar is not None:
            ax.plot(float(self._x_bar[i1]), float(self._x_bar[i2]), marker='^',
                    markersize=8, color='black', label='Sample Mean $\\bar{x}$')

        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_title(
            f"Normal-Inverse-Wishart Conjugate (n={self._n})\n"
            f"Marginal & Predictive Ellipses ({name1} vs {name2})"
        )
        ax.legend(loc='best', framealpha=0.85)

        # Autoscale limits with margin
        ax.autoscale_view()

        return ax

    def __str__(self) -> str:
        return (
            f'MVNormalInverseWishartConjugate('
            f'd={self._dim}, n={self._n}, κ₀={self._kappa_0}, ν₀={self._nu_0})'
        )

    def __repr__(self) -> str:
        return (
            f'MVNormalInverseWishartConjugate('
            f'd={self._dim}, n={self._n}, kappa_0={self._kappa_0}, nu_0={self._nu_0})'
        )
