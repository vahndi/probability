from math import sqrt
from typing import Optional, Union, Dict, List, Tuple

from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from scipy import linalg
from scipy.stats import invgamma, multivariate_normal

from probability.custom_types.external_custom_types import Array1d
from probability.distributions.continuous.inverse_gamma import InverseGamma
from probability.distributions.continuous.students_t import StudentsT
from probability.plotting import new_axes
from probability.utils import num_format


class BayesianLinearRegression(object):
    """
    Exact analytical Bayesian Linear Regression with conjugate Normal-Inverse-Gamma prior.

    Model:
        y = X β + ε,   ε ~ Normal(0, σ² I)

    Prior:
        σ² ~ InverseGamma(α₀, β₀)
        β | σ² ~ Normal(μ₀, σ² V₀)

    Posterior:
        V_n⁻¹ = V₀⁻¹ + Xᵀ X
        V_n   = (V₀⁻¹ + Xᵀ X)⁻¹
        μ_n   = V_n (V₀⁻¹ μ₀ + Xᵀ y)
        α_n   = α₀ + n / 2
        β_n   = β₀ + 1/2 * [ yᵀ y + μ₀ᵀ V₀⁻¹ μ₀ - μ_nᵀ V_n⁻¹ μ_n ]

    Marginal Posteriors:
        β_j | y, X ~ Student's t(ν = 2α_n, loc = μ_{n,j}, scale = sqrt((β_n / α_n) * (V_n)_{jj}))
        σ² | y, X  ~ InverseGamma(α_n, β_n)

    Posterior Predictive:
        y_* | x_*, y, X ~ Student's t(ν = 2α_n, loc = x_*ᵀ μ_n, scale = sqrt((β_n / α_n) * (1 + x_*ᵀ V_n x_*)))
    """

    def __init__(
            self,
            mu_n: Series,
            V_n: DataFrame,
            alpha_n: float,
            beta_n: float,
            feature_names: List[str],
            fit_intercept: bool = True,
            n_obs: int = 0,
            mu_0: Optional[Series] = None,
            V_0: Optional[DataFrame] = None,
            alpha_0: float = 0.001,
            beta_0: float = 0.001,
            X_orig: Optional[DataFrame] = None,
            y_orig: Optional[Series] = None,
    ):
        self._mu_n: Series = mu_n
        self._V_n: DataFrame = V_n
        self._alpha_n: float = float(alpha_n)
        self._beta_n: float = float(beta_n)
        self._feature_names: List[str] = feature_names
        self._fit_intercept: bool = fit_intercept
        self._n_obs: int = n_obs

        self._mu_0: Optional[Series] = mu_0
        self._V_0: Optional[DataFrame] = V_0
        self._alpha_0: float = float(alpha_0)
        self._beta_0: float = float(beta_0)

        self._X_orig: Optional[DataFrame] = X_orig
        self._y_orig: Optional[Series] = y_orig

    # region Fitting Classmethods

    @classmethod
    def fit(
            cls,
            X: Union[DataFrame, Series, ndarray, List],
            y: Union[Series, ndarray, List],
            fit_intercept: bool = True,
            alpha_0: float = 0.001,
            beta_0: float = 0.001,
            mu_0: Optional[Union[float, Array1d, Series, Dict[str, float]]] = None,
            V_0: Optional[Union[float, ndarray, DataFrame]] = None,
            prior_variance_scale: float = 1000.0,
    ) -> 'BayesianLinearRegression':
        """
        Fit a conjugate Bayesian Linear Regression model to data (X, y).

        :param X: Design matrix / predictors. Can be DataFrame, Series, or 2D array.
        :param y: Target observations.
        :param fit_intercept: Whether to include an intercept column. Default True.
        :param alpha_0: Prior shape for residual variance σ². Default 0.001 (diffuse).
        :param beta_0: Prior scale for residual variance σ². Default 0.001 (diffuse).
        :param mu_0: Prior mean vector for coefficients β. Default 0 vector.
        :param V_0: Prior covariance scale matrix V₀. Defaults to prior_variance_scale * I.
        :param prior_variance_scale: Default diagonal scale for V₀ when V₀ is None. Default 1000.0.
        """
        # 1. Format y
        if hasattr(y, 'data') and isinstance(y.data, Series):
            # Support Ratio / empirical distribution objects
            y_series = y.data.astype(float)
        elif isinstance(y, Series):
            y_series = y.astype(float)
        else:
            y_series = Series(np.asarray(y, dtype=float).ravel(), name='y')
        y_vec = y_series.values
        n = len(y_vec)

        # 2. Format X
        if isinstance(X, Series):
            X_df = DataFrame({X.name or 'x1': X})
        elif isinstance(X, DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            X_df = DataFrame(
                X_arr,
                columns=[f'x{i+1}' for i in range(X_arr.shape[1])]
            )

        X_orig = X_df.copy()

        if fit_intercept:
            if 'intercept' not in X_df.columns:
                X_df.insert(0, 'intercept', 1.0)

        feature_names = list(X_df.columns)
        p = len(feature_names)
        X_mat = X_df.values.astype(float)

        # 3. Format Prior Parameters
        if mu_0 is None:
            mu_0_vec = np.zeros(p, dtype=float)
        elif isinstance(mu_0, (int, float)):
            mu_0_vec = np.full(p, float(mu_0))
        elif isinstance(mu_0, dict):
            mu_0_vec = np.array([mu_0.get(col, 0.0) for col in feature_names], dtype=float)
        elif isinstance(mu_0, Series):
            mu_0_vec = np.array([mu_0.get(col, 0.0) for col in feature_names], dtype=float)
        else:
            mu_0_vec = np.asarray(mu_0, dtype=float).ravel()
            if len(mu_0_vec) != p:
                raise ValueError(f"mu_0 length ({len(mu_0_vec)}) must match number of features ({p})")

        mu_0_series = Series(mu_0_vec, index=feature_names, name='mu_0')

        if V_0 is None:
            V_0_mat = np.eye(p, dtype=float) * prior_variance_scale
        elif isinstance(V_0, (int, float)):
            V_0_mat = np.eye(p, dtype=float) * float(V_0)
        elif isinstance(V_0, DataFrame):
            V_0_mat = V_0.reindex(index=feature_names, columns=feature_names).values.astype(float)
        else:
            V_0_mat = np.asarray(V_0, dtype=float)

        V_0_df = DataFrame(V_0_mat, index=feature_names, columns=feature_names)

        # Invert V_0
        inv_V_0 = linalg.inv(V_0_mat)

        # 4. Compute Posterior Updates
        XtX = X_mat.T @ X_mat
        Xty = X_mat.T @ y_vec
        inv_V_n = inv_V_0 + XtX
        V_n_mat = linalg.inv(inv_V_n)

        mu_n_vec = V_n_mat @ (inv_V_0 @ mu_0_vec + Xty)
        mu_n_series = Series(mu_n_vec, index=feature_names, name='mu_n')
        V_n_df = DataFrame(V_n_mat, index=feature_names, columns=feature_names)

        alpha_n = alpha_0 + n / 2.0

        quad_y = float(y_vec.T @ y_vec)
        quad_prior = float(mu_0_vec.T @ inv_V_0 @ mu_0_vec)
        quad_post = float(mu_n_vec.T @ inv_V_n @ mu_n_vec)
        beta_diff = 0.5 * (quad_y + quad_prior - quad_post)
        beta_n = max(beta_0 + beta_diff, 1e-12)

        return cls(
            mu_n=mu_n_series,
            V_n=V_n_df,
            alpha_n=alpha_n,
            beta_n=beta_n,
            feature_names=feature_names,
            fit_intercept=fit_intercept,
            n_obs=n,
            mu_0=mu_0_series,
            V_0=V_0_df,
            alpha_0=alpha_0,
            beta_0=beta_0,
            X_orig=X_orig,
            y_orig=y_series,
        )

    # endregion

    # region Properties & Marginal Posteriors

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def alpha_n(self) -> float:
        return self._alpha_n

    @property
    def beta_n(self) -> float:
        return self._beta_n

    @property
    def mu_n(self) -> Series:
        return self._mu_n

    @property
    def V_n(self) -> DataFrame:
        return self._V_n

    @property
    def sigma_sq(self) -> InverseGamma:
        """
        Return the marginal posterior Inverse-Gamma distribution for residual variance σ².
        """
        return InverseGamma(
            alpha=self._alpha_n,
            beta=self._beta_n
        ).with_x_label('σ²').prepend_to_label('Posterior: ')

    def coef(self, name_or_idx: Union[str, int]) -> StudentsT:
        """
        Return the marginal posterior Student's t distribution for an individual coefficient.

        :param name_or_idx: Feature name (str) or index (int).
        """
        if isinstance(name_or_idx, int):
            idx = name_or_idx
            name = self._feature_names[idx]
        else:
            name = name_or_idx
            idx = self._feature_names.index(name)

        loc = float(self._mu_n.iloc[idx])
        scale_sq = (self._beta_n / self._alpha_n) * float(self._V_n.iloc[idx, idx])
        scale = sqrt(max(scale_sq, 1e-12))
        nu = 2.0 * self._alpha_n

        return StudentsT(
            nu=nu,
            mu=loc,
            sigma=scale
        ).with_x_label(f'β[{name}]').prepend_to_label(f'Posterior β[{name}]: ')

    @property
    def coefs(self) -> Dict[str, StudentsT]:
        """
        Dictionary mapping every feature name to its marginal posterior Student's t distribution.
        """
        return {name: self.coef(name) for name in self._feature_names}

    def prob_positive(self, name: str) -> float:
        """
        Posterior probability that coefficient β_j > 0.
        """
        dist = self.coef(name)
        return float(1.0 - dist.cdf().at(0.0))

    def prob_negative(self, name: str) -> float:
        """
        Posterior probability that coefficient β_j < 0.
        """
        dist = self.coef(name)
        return float(dist.cdf().at(0.0))

    def treatment_effect(
            self,
            treatment_var: str,
            control_val: float = 0.0,
            treated_val: float = 1.0
    ) -> StudentsT:
        """
        Return the posterior distribution of the treatment effect:
            TE = β_treatment * (treated_val - control_val)

        For binary treatment (0 -> 1), this equals β_treatment directly.
        """
        diff = treated_val - control_val
        dist = self.coef(treatment_var)
        if diff == 1.0:
            return dist
        return StudentsT(
            nu=dist.nu,
            mu=dist.mu * diff,
            sigma=dist.sigma * abs(diff)
        ).with_x_label(f'TE({treatment_var})').prepend_to_label(f'Treatment Effect: ')

    # endregion

    # region Predictions

    def _prepare_X_new(self, X_new: Union[DataFrame, Series, ndarray, Dict]) -> DataFrame:
        """
        Format new input features to match the design matrix.
        """
        if isinstance(X_new, Series):
            if len(self._feature_names) == 1 and not self._fit_intercept:
                df = DataFrame({self._feature_names[0]: X_new})
            else:
                # Transpose single-row series into a 1-row DataFrame
                df = DataFrame([X_new])
        elif isinstance(X_new, DataFrame):
            df = X_new.copy()
        elif isinstance(X_new, dict):
            df = DataFrame([X_new])
        else:
            arr = np.asarray(X_new, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Check if columns match expected features without intercept
            expected_no_int = [f for f in self._feature_names if f != 'intercept']
            if arr.shape[1] == len(expected_no_int):
                df = DataFrame(arr, columns=expected_no_int)
            elif arr.shape[1] == len(self._feature_names):
                df = DataFrame(arr, columns=self._feature_names)
            else:
                raise ValueError(
                    f"X_new has {arr.shape[1]} columns, expected {len(expected_no_int)}"
                )

        if self._fit_intercept and 'intercept' not in df.columns:
            df.insert(0, 'intercept', 1.0)

        # Align columns
        for col in self._feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing feature '{col}' in X_new")

        return df[self._feature_names]

    def predict_dist(
            self,
            X_new: Union[DataFrame, Series, ndarray, Dict]
    ) -> Series:
        """
        Return a Series of StudentsT distributions representing the exact
        posterior predictive distribution for each row in X_new.
        """
        X_df = self._prepare_X_new(X_new)
        X_mat = X_df.values.astype(float)
        mu_vec = self._mu_n.values
        V_mat = self._V_n.values
        nu = 2.0 * self._alpha_n
        ratio = self._beta_n / self._alpha_n

        dists = []
        for i in range(len(X_df)):
            x_i = X_mat[i]
            loc = float(x_i @ mu_vec)
            var_scale = 1.0 + float(x_i @ V_mat @ x_i)
            scale = sqrt(max(ratio * var_scale, 1e-12))
            dists.append(
                StudentsT(nu=nu, mu=loc, sigma=scale).with_x_label('y_*')
            )

        return Series(dists, index=X_df.index, name='predictive_dist')

    def predict(
            self,
            X_new: Union[DataFrame, Series, ndarray, Dict]
    ) -> Series:
        """
        Return point predictions (posterior expected values E[y_*]).
        """
        X_df = self._prepare_X_new(X_new)
        y_hat = X_df.values.astype(float) @ self._mu_n.values
        return Series(y_hat, index=X_df.index, name='prediction')

    def predict_interval(
            self,
            X_new: Union[DataFrame, Series, ndarray, Dict],
            hdi: float = 0.95
    ) -> DataFrame:
        """
        Return point predictions along with lower and upper credible intervals (HDIs).
        """
        dists = self.predict_dist(X_new)
        records = []
        for dist in dists:
            low, high = dist.hdi(hdi)
            records.append({
                'mean': dist.mu,
                f'hdi_{int(hdi * 100)}_lower': low,
                f'hdi_{int(hdi * 100)}_upper': high
            })
        return DataFrame(records, index=dists.index)

    def sample_posterior(self, num_samples: int = 1000) -> Tuple[DataFrame, np.ndarray]:
        """
        Draw joint (β, σ²) samples from the posterior.

        :return: (beta_samples_df, sigma_sq_samples)
        """
        sig_sq_samples = invgamma.rvs(
            self._alpha_n, scale=self._beta_n, size=num_samples
        )
        beta_samples = []
        V_mat = self._V_n.values
        mu_vec = self._mu_n.values

        for s in sig_sq_samples:
            cov = s * V_mat
            beta_draw = multivariate_normal.rvs(mean=mu_vec, cov=cov)
            beta_samples.append(beta_draw)

        beta_df = DataFrame(
            beta_samples, columns=self._feature_names
        )
        return beta_df, sig_sq_samples

    # endregion

    # region Summary and Plotting

    def summary(self, hdi: float = 0.95) -> DataFrame:
        """
        Return a summary DataFrame containing mean, standard deviation,
        Highest Density Intervals (HDIs), and directional posterior probabilities
        for all coefficients and residual variance.
        """
        records = []
        hdi_pct = int(hdi * 100)
        for name in self._feature_names:
            c = self.coef(name)
            low, high = c.hdi(hdi)
            records.append({
                'parameter': f'beta[{name}]',
                'mean': c.mu,
                'std': c.sigma,
                f'hdi_{hdi_pct}_lower': low,
                f'hdi_{hdi_pct}_upper': high,
                'P(>0)': float(1.0 - c.cdf().at(0.0)),
                'P(<0)': float(c.cdf().at(0.0)),
            })

        # Add residual variance
        sig = self.sigma_sq
        sig_low, sig_high = sig.hdi(hdi)
        records.append({
            'parameter': 'sigma_sq',
            'mean': self._beta_n / (self._alpha_n - 1.0) if self._alpha_n > 1 else np.nan,
            'std': np.nan,
            f'hdi_{hdi_pct}_lower': sig_low,
            f'hdi_{hdi_pct}_upper': sig_high,
            'P(>0)': 1.0,
            'P(<0)': 0.0,
        })

        return DataFrame(records).set_index('parameter')

    def plot_coefficients(
            self,
            hdi: float = 0.95,
            exclude_intercept: bool = False,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot a forest plot of posterior coefficients with their means and HDIs.
        """
        ax = ax or new_axes()
        names = [f for f in self._feature_names if not (exclude_intercept and f == 'intercept')]
        y_positions = np.arange(len(names))

        means = []
        lows = []
        highs = []
        for name in names:
            c = self.coef(name)
            low, high = c.hdi(hdi)
            means.append(c.mu)
            lows.append(low)
            highs.append(high)

        means = np.array(means)
        lows = np.array(lows)
        highs = np.array(highs)

        # Plot error bars
        xerr = np.vstack([means - lows, highs - means])
        ax.errorbar(
            means, y_positions, xerr=xerr,
            fmt='o', color='darkblue', ecolor='teal',
            elinewidth=2, capsize=4, capthick=1.5
        )

        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'Posterior Coefficients ({int(hdi * 100)}% HDI)')
        ax.grid(True, linestyle=':', alpha=0.5)

        return ax

    def plot_fit(
            self,
            x_var: Optional[str] = None,
            hdi: float = 0.95,
            n_grid: int = 100,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        For models with a primary continuous predictor, plot observed data points,
        posterior regression line, and shaded HDI predictive interval.
        """
        if self._X_orig is None or self._y_orig is None:
            raise ValueError("plot_fit requires original data from fit()")

        ax = ax or new_axes()
        predictors = [f for f in self._X_orig.columns if f != 'intercept']
        if x_var is None:
            if len(predictors) != 1:
                raise ValueError(
                    f"Please specify x_var from available predictors: {predictors}"
                )
            x_var = predictors[0]

        x_vals = self._X_orig[x_var].values
        y_vals = self._y_orig.values

        # Scatter plot of raw data
        ax.scatter(x_vals, y_vals, color='darkgray', alpha=0.7, label='Observed Data')

        # Grid of x values
        x_min, x_max = x_vals.min(), x_vals.max()
        x_span = x_max - x_min
        grid_x = np.linspace(x_min - 0.05 * x_span, x_max + 0.05 * x_span, n_grid)

        # Build grid DataFrame for predictions
        grid_dict = {x_var: grid_x}
        for col in predictors:
            if col != x_var:
                grid_dict[col] = float(self._X_orig[col].mean())
        grid_df = DataFrame(grid_dict)

        # Get intervals
        intervals = self.predict_interval(grid_df, hdi=hdi)
        hdi_pct = int(hdi * 100)
        low_col = f'hdi_{hdi_pct}_lower'
        high_col = f'hdi_{hdi_pct}_upper'

        ax.plot(grid_x, intervals['mean'], color='navy', linewidth=2, label='Posterior Mean')
        ax.fill_between(
            grid_x, intervals[low_col], intervals[high_col],
            color='skyblue', alpha=0.4, label=f'{hdi_pct}% Predictive HDI'
        )

        ax.set_xlabel(x_var)
        ax.set_ylabel(self._y_orig.name or 'y')
        ax.set_title(f'Bayesian Linear Regression Fit: {self._y_orig.name or "y"} ~ {x_var}')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

        return ax

    # endregion

    def __str__(self) -> str:
        coef_str = ', '.join([
            f'{name}={num_format(self._mu_n[name], 3)}'
            for name in self._feature_names
        ])
        return f'BayesianLinearRegression(n={self._n_obs}, β=[{coef_str}])'

    def __repr__(self) -> str:
        return self.__str__()
