from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import norm

from probability.distributions.continuous.normal import Normal
from probability.distributions.multivariate.mv_normal import MVNormal
from probability.plotting import new_axes


class LaplaceApproximation(object):
    """
    Quadratic / Laplace approximation for Bayesian posterior inference.

    Approximates an arbitrary continuous posterior distribution p(θ | D)
    around the Maximum A Posteriori (MAP) estimate θ̂ with a Gaussian distribution:
        p(θ | D) ≈ N(θ̂, Σ = H⁻¹)
    where H is the observed information matrix (Hessian of the negative log-posterior
    evaluated at θ̂).

    Supports:
      - Fitting via scipy.optimize on arbitrary log-posterior or prior + likelihood functions
      - Support for parameter bounds (e.g. variances > 0)
      - Robust central finite-difference Hessian and covariance matrix estimation
      - Component marginals returning 1D `Normal` objects
      - Joint posterior returning `Normal` (1D) or `MVNormal` (multivariate)
      - Exact parameter summaries and Highest Density Intervals (HDIs)
      - Joint posterior sampling
      - Forest plots and marginal visualizations
    """

    def __init__(
            self,
            mode: Sequence[float],
            cov: ndarray,
            param_names: Optional[Sequence[str]] = None,
            opt_result: Optional[OptimizeResult] = None
    ):
        """
        Direct constructor for LaplaceApproximation. Typically instantiated via .fit().

        :param mode: Parameter vector at the MAP mode.
        :param cov: 2D covariance matrix Σ = H⁻¹.
        :param param_names: Optional list of parameter names.
        :param opt_result: Scipy optimization result object.
        """
        mode_arr = np.asarray(mode, dtype=float).ravel()
        cov_arr = np.asarray(cov, dtype=float)
        d = len(mode_arr)

        if cov_arr.shape != (d, d):
            raise ValueError(f"Covariance shape {cov_arr.shape} must match parameter dimension ({d}, {d}).")

        if param_names is None:
            if d == 1:
                names = ['theta']
            else:
                names = [f'theta_{i+1}' for i in range(d)]
        else:
            names = list(param_names)
            if len(names) != d:
                raise ValueError(f"param_names length ({len(names)}) must match parameter dimension ({d}).")

        self._param_names: List[str] = names
        self._mode: Series = Series(mode_arr, index=self._param_names, name='mode')
        self._cov: DataFrame = DataFrame(cov_arr, index=self._param_names, columns=self._param_names)
        variances = np.diag(cov_arr)
        # Avoid negative variances due to float precision
        variances = np.maximum(variances, 0.0)
        self._std: Series = Series(np.sqrt(variances), index=self._param_names, name='std')
        self._opt_result: Optional[OptimizeResult] = opt_result

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def param_names(self) -> List[str]:
        return list(self._param_names)

    @property
    def ndim(self) -> int:
        return len(self._param_names)

    @property
    def mode(self) -> Series:
        """MAP mode / mean vector of parameters."""
        return self._mode.copy()

    @property
    def mean(self) -> Series:
        """Mean vector (identical to mode under Gaussian approximation)."""
        return self._mode.copy()

    @property
    def cov(self) -> DataFrame:
        """Posterior covariance matrix Σ = H⁻¹."""
        return self._cov.copy()

    @property
    def std(self) -> Series:
        """Standard errors of parameters (square root of diagonal of covariance)."""
        return self._std.copy()

    @property
    def opt_result(self) -> Optional[OptimizeResult]:
        return self._opt_result

    @property
    def success(self) -> bool:
        return bool(self._opt_result.success) if self._opt_result is not None else True

    # -------------------------------------------------------------------------
    # Fitting Classmethods
    # -------------------------------------------------------------------------

    @classmethod
    def fit(
            cls,
            log_posterior_fn: Callable[[ndarray], float],
            start_params: Union[Sequence[float], Dict[str, float]],
            param_names: Optional[Sequence[str]] = None,
            bounds: Optional[Union[Sequence[Tuple[Optional[float], Optional[float]]], Dict[str, Tuple[Optional[float], Optional[float]]]]] = None,
            method: Optional[str] = None,
            h: float = 1e-4,
            **minimize_kwargs
    ) -> 'LaplaceApproximation':
        """
        Fit a Laplace (quadratic) approximation to an unnormalized log-posterior function.

        :param log_posterior_fn: Function taking parameter vector theta returning log p(theta | data).
        :param start_params: Initial parameter guess (sequence or dict mapping param_name -> value).
        :param param_names: Optional parameter names (inferred if start_params is dict).
        :param bounds: Optional parameter bounds (sequence of (low, high) or dict).
        :param method: Scipy minimize algorithm (e.g. 'L-BFGS-B', 'BFGS', 'Nelder-Mead').
        :param h: Finite-difference step size for numerical Hessian calculation.
        :param minimize_kwargs: Additional kwargs passed to scipy.optimize.minimize.
        """
        # Parse start_params and names
        if isinstance(start_params, dict):
            names = list(start_params.keys())
            x0 = np.array([start_params[k] for k in names], dtype=float)
        else:
            x0 = np.asarray(start_params, dtype=float)
            names = list(param_names) if param_names is not None else [f'theta_{i+1}' for i in range(len(x0))]

        # Parse bounds
        opt_bounds = None
        if bounds is not None:
            if isinstance(bounds, dict):
                opt_bounds = [bounds.get(k, (None, None)) for k in names]
            else:
                opt_bounds = list(bounds)

        # Negative log-posterior objective
        def objective(theta):
            val = log_posterior_fn(theta)
            if np.isnan(val) or np.isneginf(val):
                return 1e12
            return -float(val)

        # Select method
        if method is None:
            method = 'L-BFGS-B' if opt_bounds is not None else 'BFGS'

        res = minimize(
            fun=objective,
            x0=x0,
            bounds=opt_bounds,
            method=method,
            **minimize_kwargs
        )

        mode_arr = res.x
        d = len(mode_arr)

        # Compute numerical Hessian of objective (negative log posterior) at mode
        H = cls._compute_hessian(objective, mode_arr, h=h)

        # Invert Hessian to get covariance matrix
        cov = cls._invert_hessian(H)

        return cls(mode=mode_arr, cov=cov, param_names=names, opt_result=res)

    @classmethod
    def fit_model(
            cls,
            log_prior_fn: Callable[[ndarray], float],
            log_likelihood_fn: Callable[[ndarray, Any], float],
            start_params: Union[Sequence[float], Dict[str, float]],
            data: Any = None,
            param_names: Optional[Sequence[str]] = None,
            bounds: Optional[Union[Sequence[Tuple[Optional[float], Optional[float]]], Dict[str, Tuple[Optional[float], Optional[float]]]]] = None,
            **kwargs
    ) -> 'LaplaceApproximation':
        """
        Fit a Laplace approximation given separate log-prior and log-likelihood functions:
            log_posterior(theta) = log_prior(theta) + log_likelihood(theta, data).
        """
        def log_post(theta):
            lp = log_prior_fn(theta)
            if np.isneginf(lp):
                return -np.inf
            ll = log_likelihood_fn(theta, data) if data is not None else log_likelihood_fn(theta)
            return lp + ll

        return cls.fit(
            log_posterior_fn=log_post,
            start_params=start_params,
            param_names=param_names,
            bounds=bounds,
            **kwargs
        )

    # -------------------------------------------------------------------------
    # Numerical Hessian & Inversion
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_hessian(fn: Callable[[ndarray], float], x: ndarray, h: float = 1e-4) -> ndarray:
        """
        Compute Hessian matrix of scalar function fn at point x using central finite differences.
        """
        n = len(x)
        H = np.zeros((n, n), dtype=float)
        f0 = fn(x)

        # Steps scaled by parameter magnitude
        steps = np.maximum(h, np.abs(x) * h)

        for i in range(n):
            ei = np.zeros(n)
            ei[i] = steps[i]

            # Diagonal: (f(x + h) - 2*f(x) + f(x - h)) / h^2
            f_plus = fn(x + ei)
            f_minus = fn(x - ei)
            H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = steps[j]

                # Off-diagonal: (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*hi*hj)
                f_pp = fn(x + ei + ej)
                f_pm = fn(x + ei - ej)
                f_mp = fn(x - ei + ej)
                f_mm = fn(x - ei - ej)

                val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
                H[i, j] = val
                H[j, i] = val

        # Symmetrize
        return 0.5 * (H + H.T)

    @staticmethod
    def _invert_hessian(H: ndarray) -> ndarray:
        """
        Safely invert the Hessian matrix, ensuring positive semi-definiteness.
        """
        d = H.shape[0]
        # Eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(H)
            # If any eigenvalues are non-positive or very small, regularize
            min_eig = 1e-8
            if np.any(eigvals <= min_eig):
                eigvals = np.maximum(eigvals, min_eig)
            inv_eigvals = 1.0 / eigvals
            cov = eigvecs @ np.diag(inv_eigvals) @ eigvecs.T
        except Exception:
            # Fallback to pseudo-inverse with diagonal jitter
            H_reg = H + np.eye(d) * 1e-6
            cov = np.linalg.pinv(H_reg)

        return 0.5 * (cov + cov.T)

    # -------------------------------------------------------------------------
    # Posterior Distributions & Marginals
    # -------------------------------------------------------------------------

    def marginal(self, param: Union[str, int]) -> Normal:
        """
        Return the 1D marginal Gaussian distribution for a specific parameter.

        :param param: Parameter name (str) or index (int).
        :return: Normal(mu=..., sigma=...) distribution object.
        """
        if isinstance(param, int):
            param_name = self._param_names[param]
        else:
            param_name = str(param)
            if param_name not in self._param_names:
                raise ValueError(f"Parameter '{param_name}' not found. Available: {self._param_names}")

        mu = float(self._mode[param_name])
        sigma = float(self._std[param_name])
        norm_dist = Normal(mu=mu, sigma=sigma)
        norm_dist.with_label(param_name)
        return norm_dist

    @property
    def posterior(self) -> Union[Normal, MVNormal]:
        """
        Return the joint Gaussian posterior distribution.
        Returns `Normal` for 1D models, or `MVNormal` for multi-parameter models.
        """
        if self.ndim == 1:
            return self.marginal(0)
        else:
            return MVNormal(mu=self._mode.to_dict(), sigma=self._cov.values)

    # -------------------------------------------------------------------------
    # Summary & Inference
    # -------------------------------------------------------------------------

    def summary(self, hdi: float = 0.95) -> DataFrame:
        """
        Generate a tidy summary table with MAP mean, standard error, and HDI bounds.

        :param hdi: Highest density interval probability (default 0.95).
        """
        alpha = 1.0 - hdi
        z = norm.ppf(1.0 - alpha / 2.0)

        df = DataFrame(index=self._param_names)
        df['mean'] = self._mode
        df['std'] = self._std
        df[f'{hdi:.0%} HDI lower'] = self._mode - z * self._std
        df[f'{hdi:.0%} HDI upper'] = self._mode + z * self._std
        return df

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> DataFrame:
        """
        Draw samples from the joint Gaussian posterior approximation.

        :param size: Number of draws.
        :param seed: Optional random seed.
        :return: DataFrame of draws of shape (size, ndim).
        """
        if seed is not None:
            np.random.seed(seed)
        draws = np.random.multivariate_normal(
            mean=self._mode.values,
            cov=self._cov.values,
            size=size
        )
        return DataFrame(draws, columns=self._param_names)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_forest(
            self,
            hdi: float = 0.95,
            title: Optional[str] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Forest plot displaying MAP point estimates and horizontal HDI error bars.
        """
        ax = ax or new_axes()
        alpha = 1.0 - hdi
        z = norm.ppf(1.0 - alpha / 2.0)

        y_positions = np.arange(len(self._param_names))
        means = self._mode.values
        stds = self._std.values
        lower_errs = z * stds
        upper_errs = z * stds

        ax.errorbar(
            means, y_positions,
            xerr=[lower_errs, upper_errs],
            fmt='o',
            color=kwargs.get('color', 'tab:blue'),
            ecolor=kwargs.get('ecolor', 'tab:blue'),
            elinewidth=kwargs.get('elinewidth', 2.0),
            capsize=kwargs.get('capsize', 4.0),
            markersize=kwargs.get('markersize', 6.0)
        )

        ax.axvline(0.0, color='red', linestyle='--', alpha=0.6, linewidth=1.0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(self._param_names)
        ax.set_xlabel('Parameter Estimate')
        ax.set_title(title or f"Laplace Approximation: Parameter HDIs ({hdi:.0%})")
        ax.invert_yaxis()
        return ax

    def plot_marginal(
            self,
            param: Union[str, int],
            hdi: float = 0.95,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot the 1D marginal posterior density curve with shaded HDI.
        """
        ax = ax or new_axes()
        dist = self.marginal(param)
        lbl = str(param) if isinstance(param, str) else self._param_names[param]
        mu, sigma = dist.mu, dist.sigma

        x_vals = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 300)
        y_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
        ax.plot(x_vals, y_vals, label=f"{lbl} ~ N({mu:.2f}, {sigma:.2f})", color=kwargs.get('color', 'tab:blue'))

        # Shade HDI
        alpha = 1.0 - hdi
        z = norm.ppf(1.0 - alpha / 2.0)
        low, high = mu - z * sigma, mu + z * sigma
        mask = (x_vals >= low) & (x_vals <= high)
        ax.fill_between(
            x_vals[mask], 0, y_vals[mask],
            alpha=kwargs.get('alpha', 0.3),
            color=kwargs.get('color', 'tab:blue'),
            label=f"{hdi:.0%} HDI [{low:.2f}, {high:.2f}]"
        )

        ax.set_xlabel(lbl)
        ax.set_ylabel('Density')
        ax.set_title(f"Marginal Posterior: {lbl}")
        ax.legend(loc='best')
        return ax

    def __str__(self) -> str:
        items = [f"{k}={self._mode[k]:.3f}±{self._std[k]:.3f}" for k in self._param_names]
        return f"LaplaceApproximation({', '.join(items)})"

    def __repr__(self) -> str:
        return self.__str__()


# Convenience Aliases
QuadraticApproximation = LaplaceApproximation
Quap = LaplaceApproximation
