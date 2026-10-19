from typing import Any, Callable, Optional, Sequence, Tuple

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from probability.distributions.grid.grid_distribution_1d import GridDistribution1d
from probability.plotting import new_axes


class GridDistribution2d(object):
    """
    2-Dimensional discrete grid approximation for joint Bayesian inference and distributions.

    Represents a joint probability distribution over two ordered 1D support grids:
    X: x₁ < x₂ < ... < x_m
    Y: y₁ < y₂ < ... < y_n
    with joint normalized probability mass matrix P of shape (m, n),
    where P[i, j] = P(X = x_i, Y = y_j) and sum(P) = 1.0.

    Supports:
      - Direct probability matrix initialization and uniform presets
      - Construction from independent 1D distributions (outer product)
      - Construction from arbitrary joint density functions
      - Non-conjugate Bayesian sequential updating: posterior ∝ prior × likelihood
      - Dedicated 2D updates for Normal parameters (μ, σ) and Logistic regression (α, β)
      - Component marginals P(X) and P(Y) returning GridDistribution1d
      - Slice conditionals P(X | Y = y) and P(Y | X = x) returning GridDistribution1d
      - Exact joint statistics: means, modes, covariance matrix, correlation
      - 2D Highest Density Region (HDR) masks
      - Native Matplotlib contour, filled contour, and heatmap visualizations
    """

    def __init__(
            self,
            x_grid: Sequence[float],
            y_grid: Sequence[float],
            probs: Optional[Sequence[Sequence[float]]] = None,
            x_name: str = 'X',
            y_name: str = 'Y'
    ):
        """
        Create a new 2D Grid Distribution.

        :param x_grid: Sequence of grid points for variable X (numeric, length >= 2).
        :param y_grid: Sequence of grid points for variable Y (numeric, length >= 2).
        :param probs: Optional 2D array-like of shape (len(x_grid), len(y_grid)).
                      If None, initialized to uniform probabilities.
        :param x_name: Name for variable X.
        :param y_name: Name for variable Y.
        """
        x_arr = np.asarray(x_grid, dtype=float)
        y_arr = np.asarray(y_grid, dtype=float)

        if x_arr.ndim != 1 or len(x_arr) < 2:
            raise ValueError("x_grid must be a 1D sequence containing at least 2 points.")
        if y_arr.ndim != 1 or len(y_arr) < 2:
            raise ValueError("y_grid must be a 1D sequence containing at least 2 points.")

        # Ensure sorted grids
        x_sort = np.argsort(x_arr)
        y_sort = np.argsort(y_arr)
        self._x_grid: ndarray = x_arr[x_sort]
        self._y_grid: ndarray = y_arr[y_sort]

        nx = len(self._x_grid)
        ny = len(self._y_grid)

        if probs is None:
            self._probs: ndarray = np.full((nx, ny), 1.0 / (nx * ny), dtype=float)
        else:
            p_arr = np.asarray(probs, dtype=float)
            if p_arr.shape != (len(x_arr), len(y_arr)):
                raise ValueError(
                    f"probs shape {p_arr.shape} must match (len(x_grid), len(y_grid)) = ({len(x_arr)}, {len(y_arr)})."
                )
            # Reorder rows and cols to match sorted grids
            p_arr = p_arr[x_sort, :][:, y_sort]
            if np.any(p_arr < 0):
                raise ValueError("Probabilities must be non-negative.")
            total = p_arr.sum()
            if total <= 0:
                raise ValueError("Sum of probabilities must be positive.")
            self._probs = p_arr / total

        self._x_name: str = x_name
        self._y_name: str = y_name

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def uniform(
            cls,
            x_low: float,
            x_high: float,
            x_num: int = 100,
            y_low: float = 0.0,
            y_high: float = 1.0,
            y_num: int = 100,
            x_name: str = 'X',
            y_name: str = 'Y'
    ) -> 'GridDistribution2d':
        """
        Create a uniform 2D distribution over the rectangular grid [x_low, x_high] × [y_low, y_high].
        """
        if x_high <= x_low:
            raise ValueError("x_high must be greater than x_low.")
        if y_high <= y_low:
            raise ValueError("y_high must be greater than y_low.")
        x_grid = np.linspace(x_low, x_high, x_num)
        y_grid = np.linspace(y_low, y_high, y_num)
        return cls(x_grid=x_grid, y_grid=y_grid, probs=None, x_name=x_name, y_name=y_name)

    @classmethod
    def from_independent(
            cls,
            dist_x: GridDistribution1d,
            dist_y: GridDistribution1d,
            x_name: Optional[str] = None,
            y_name: Optional[str] = None
    ) -> 'GridDistribution2d':
        """
        Create a 2D joint distribution assuming X and Y are independent: P(X, Y) = P(X) × P(Y).
        """
        probs_2d = np.outer(dist_x.probs, dist_y.probs)
        return cls(
            x_grid=dist_x.grid,
            y_grid=dist_y.grid,
            probs=probs_2d,
            x_name=x_name or dist_x.name or 'X',
            y_name=y_name or dist_y.name or 'Y'
        )

    @classmethod
    def from_function(
            cls,
            fn: Callable[[ndarray, ndarray], ndarray],
            x_grid: Sequence[float],
            y_grid: Sequence[float],
            x_name: str = 'X',
            y_name: str = 'Y'
    ) -> 'GridDistribution2d':
        """
        Construct a 2D grid distribution from a probability density function fn(X_mesh, Y_mesh) or fn(x, y).

        :param fn: Function taking 2D meshgrids (or 1D coordinates) and returning density values.
        :param x_grid: Sequence of X grid coordinates.
        :param y_grid: Sequence of Y grid coordinates.
        :param x_name: Name for X.
        :param y_name: Name for Y.
        """
        x_arr = np.asarray(x_grid, dtype=float)
        y_arr = np.asarray(y_grid, dtype=float)
        xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')
        try:
            raw_vals = fn(xx, yy)
        except (TypeError, ValueError):
            # Fall back to element-wise evaluation
            raw_vals = np.zeros_like(xx)
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    raw_vals[i, j] = fn(xx[i, j], yy[i, j])

        raw_vals = np.asarray(raw_vals, dtype=float)
        raw_vals = np.clip(raw_vals, 0.0, None)
        total = raw_vals.sum()
        if total <= 0:
            raise ValueError("Density function evaluates to 0 over the entire grid.")
        return cls(x_grid=x_arr, y_grid=y_arr, probs=raw_vals / total, x_name=x_name, y_name=y_name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def x_grid(self) -> ndarray:
        return self._x_grid.copy()

    @property
    def y_grid(self) -> ndarray:
        return self._y_grid.copy()

    @property
    def probs(self) -> ndarray:
        return self._probs.copy()

    @property
    def shape(self) -> Tuple[int, int]:
        return self._probs.shape

    @property
    def x_name(self) -> str:
        return self._x_name

    @property
    def y_name(self) -> str:
        return self._y_name

    @property
    def meshgrid(self) -> Tuple[ndarray, ndarray]:
        """
        Return (X_mesh, Y_mesh) using 'ij' indexing (shape matches probs).
        """
        return np.meshgrid(self._x_grid, self._y_grid, indexing='ij')

    # -------------------------------------------------------------------------
    # Marginals & Conditionals
    # -------------------------------------------------------------------------

    def marginal_x(self) -> GridDistribution1d:
        """
        Compute the 1D marginal distribution P(X) = sum_y P(X, Y).
        """
        p_x = self._probs.sum(axis=1)
        return GridDistribution1d(grid=self._x_grid, probs=p_x, name=self._x_name)

    def marginal_y(self) -> GridDistribution1d:
        """
        Compute the 1D marginal distribution P(Y) = sum_x P(X, Y).
        """
        p_y = self._probs.sum(axis=0)
        return GridDistribution1d(grid=self._y_grid, probs=p_y, name=self._y_name)

    def conditional_x(self, given_y: float) -> GridDistribution1d:
        """
        Compute the conditional distribution P(X | Y = given_y).
        Finds the nearest grid point y_j to given_y and conditions on that slice.
        """
        j = int(np.argmin(np.abs(self._y_grid - given_y)))
        slice_probs = self._probs[:, j]
        total = slice_probs.sum()
        if total <= 0:
            raise ValueError(f"Conditional slice at Y={self._y_grid[j]} has 0 probability mass.")
        return GridDistribution1d(
            grid=self._x_grid,
            probs=slice_probs / total,
            name=f"{self._x_name} | {self._y_name}={self._y_grid[j]:.3f}"
        )

    def conditional_y(self, given_x: float) -> GridDistribution1d:
        """
        Compute the conditional distribution P(Y | X = given_x).
        Finds the nearest grid point x_i to given_x and conditions on that slice.
        """
        i = int(np.argmin(np.abs(self._x_grid - given_x)))
        slice_probs = self._probs[i, :]
        total = slice_probs.sum()
        if total <= 0:
            raise ValueError(f"Conditional slice at X={self._x_grid[i]} has 0 probability mass.")
        return GridDistribution1d(
            grid=self._y_grid,
            probs=slice_probs / total,
            name=f"{self._y_name} | {self._x_name}={self._x_grid[i]:.3f}"
        )

    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------

    def mean(self) -> Tuple[float, float]:
        """
        Calculate expected values (E[X], E[Y]).
        """
        mu_x = float(self.marginal_x().mean())
        mu_y = float(self.marginal_y().mean())
        return mu_x, mu_y

    def mode(self) -> Tuple[float, float]:
        """
        Return the coordinates (x_mode, y_mode) that maximize joint probability P(X, Y).
        """
        max_idx = np.unravel_index(np.argmax(self._probs), self._probs.shape)
        return float(self._x_grid[max_idx[0]]), float(self._y_grid[max_idx[1]])

    def cov(self) -> ndarray:
        """
        Compute the 2x2 joint covariance matrix:
        [[Var(X), Cov(X, Y)],
         [Cov(X, Y), Var(Y)]]
        """
        mu_x, mu_y = self.mean()
        xx, yy = self.meshgrid
        e_xy = float(np.sum(xx * yy * self._probs))
        cov_xy = e_xy - (mu_x * mu_y)
        var_x = float(self.marginal_x().var())
        var_y = float(self.marginal_y().var())
        return np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=float)

    def corr(self) -> float:
        """
        Compute the Pearson correlation coefficient between X and Y under the joint distribution.
        """
        c = self.cov()
        denom = np.sqrt(c[0, 0] * c[1, 1])
        if denom == 0:
            return 0.0
        return float(np.clip(c[0, 1] / denom, -1.0, 1.0))

    def sample(self, size: int = 1, seed: Optional[int] = None) -> ndarray:
        """
        Draw samples from the 2D joint distribution.

        :param size: Number of sample pairs to draw.
        :param seed: Optional random seed.
        :return: Array of shape (size, 2) containing [X, Y] pairs.
        """
        if seed is not None:
            np.random.seed(seed)
        flat_probs = self._probs.ravel()
        indices = np.random.choice(len(flat_probs), size=size, p=flat_probs)
        x_idx, y_idx = np.unravel_index(indices, self._probs.shape)
        return np.column_stack([self._x_grid[x_idx], self._y_grid[y_idx]])

    def rvs(self, size: int = 1, seed: Optional[int] = None) -> ndarray:
        """Alias for sample()."""
        return self.sample(size=size, seed=seed)

    # -------------------------------------------------------------------------
    # Highest Density Region (HDR)
    # -------------------------------------------------------------------------

    def hdr_mask(self, prob: float = 0.95) -> ndarray:
        """
        Compute a 2D boolean mask of shape (nx, ny) representing the Highest Density Region (HDR)
        covering at least `prob` total probability mass.

        :param prob: Probability mass to cover (e.g. 0.95).
        :return: Boolean 2D array where True indicates cells in the HDR.
        """
        if not 0.0 < prob < 1.0:
            raise ValueError("prob must be strictly between 0 and 1.")
        flat_p = self._probs.ravel()
        sort_order = np.argsort(flat_p)[::-1]
        cumsum = np.cumsum(flat_p[sort_order])
        cutoff_idx = int(np.searchsorted(cumsum, prob))
        cutoff_idx = min(cutoff_idx, len(flat_p) - 1)
        threshold = flat_p[sort_order[cutoff_idx]]

        return self._probs >= threshold

    # -------------------------------------------------------------------------
    # Bayesian Updating
    # -------------------------------------------------------------------------

    def update_likelihood(self, likelihood_matrix: Sequence[Sequence[float]]) -> 'GridDistribution2d':
        """
        Update the 2D prior with a precomputed likelihood matrix:
        posterior ∝ prior × likelihood.

        :param likelihood_matrix: 2D array-like matching shape (len(x_grid), len(y_grid)).
        """
        lik = np.asarray(likelihood_matrix, dtype=float)
        if lik.shape != self._probs.shape:
            raise ValueError(
                f"Likelihood matrix shape {lik.shape} must match grid shape {self._probs.shape}."
            )
        if np.any(lik < 0):
            raise ValueError("Likelihood values must be non-negative.")

        # Log-space multiplication for numerical safety
        with np.errstate(divide='ignore'):
            log_prior = np.where(self._probs > 0, np.log(self._probs), -np.inf)
            log_lik = np.where(lik > 0, np.log(lik), -np.inf)
        log_unnorm = log_prior + log_lik

        max_log = np.max(log_unnorm)
        if np.isneginf(max_log):
            raise ValueError("Updated posterior has zero probability across the entire grid.")

        unnorm = np.exp(log_unnorm - max_log)
        post_probs = unnorm / unnorm.sum()

        return GridDistribution2d(
            x_grid=self._x_grid,
            y_grid=self._y_grid,
            probs=post_probs,
            x_name=self._x_name,
            y_name=self._y_name
        )

    def update(
            self,
            likelihood_fn: Callable[[ndarray, ndarray, Any], ndarray],
            data: Any
    ) -> 'GridDistribution2d':
        """
        Update with an arbitrary likelihood function:
        likelihood_fn(X_mesh, Y_mesh, data) -> 2D likelihood matrix.

        :param likelihood_fn: Function taking (X_mesh, Y_mesh, data) returning non-negative likelihoods.
        :param data: Observed data passed to likelihood_fn.
        """
        xx, yy = self.meshgrid
        lik = likelihood_fn(xx, yy, data)
        return self.update_likelihood(lik)

    def update_normal(self, data: Sequence[float]) -> 'GridDistribution2d':
        """
        Bayesian update for Normal data where X = μ (mean) and Y = σ (standard deviation).
        Log-likelihood: sum_k [ -ln(σ) - (d_k - μ)² / (2σ²) ].

        :param data: Sequence of observed numeric data points.
        """
        d_arr = np.asarray(data, dtype=float)
        if len(d_arr) == 0:
            return self

        xx, yy = self.meshgrid  # xx is mu, yy is sigma
        # Ensure sigma > 0
        if np.any(yy <= 0):
            raise ValueError("y_grid (sigma) must be strictly positive.")

        n = len(d_arr)
        # Vectorized log-likelihood computation
        # For each (mu, sigma), log_lik = -n * ln(sigma) - sum((d - mu)^2) / (2 * sigma^2)
        # sum((d - mu)^2) = sum(d^2) - 2*mu*sum(d) + n*mu^2
        sum_d = d_arr.sum()
        sum_d2 = (d_arr ** 2).sum()

        ss_error = sum_d2 - 2 * xx * sum_d + n * (xx ** 2)
        log_lik = -n * np.log(yy) - ss_error / (2 * (yy ** 2))

        # Add to log prior
        with np.errstate(divide='ignore'):
            log_prior = np.where(self._probs > 0, np.log(self._probs), -np.inf)
        log_post = log_prior + log_lik
        log_post -= np.max(log_post)
        unnorm = np.exp(log_post)
        post_probs = unnorm / unnorm.sum()

        return GridDistribution2d(
            x_grid=self._x_grid,
            y_grid=self._y_grid,
            probs=post_probs,
            x_name=self._x_name or 'μ',
            y_name=self._y_name or 'σ'
        )

    def update_logistic_regression(
            self,
            x_data: Sequence[float],
            y_binary: Sequence[int]
    ) -> 'GridDistribution2d':
        """
        Bayesian update for simple Logistic Regression where X = α (intercept) and Y = β (slope):
        P(y_k = 1 | x_k) = 1 / (1 + exp(-(α + β x_k))).

        :param x_data: Predictor values x_k.
        :param y_binary: Binary outcomes y_k in {0, 1}.
        """
        xs = np.asarray(x_data, dtype=float)
        ys = np.asarray(y_binary, dtype=int)
        if len(xs) != len(ys):
            raise ValueError("x_data and y_binary must have the same length.")

        xx, yy = self.meshgrid  # shape (nx, ny)
        log_lik = np.zeros_like(xx)

        for x_val, y_val in zip(xs, ys):
            # linear predictor eta = alpha + beta * x
            eta = xx + yy * x_val
            # log(sigmoid(eta)) = -log(1 + exp(-eta))
            # log(1 - sigmoid(eta)) = -eta - log(1 + exp(-eta))
            # numerically stable log(1 + exp(u)) = np.logaddexp(0, u)
            if y_val == 1:
                log_lik -= np.logaddexp(0, -eta)
            else:
                log_lik -= np.logaddexp(0, eta)

        with np.errstate(divide='ignore'):
            log_prior = np.where(self._probs > 0, np.log(self._probs), -np.inf)
        log_post = log_prior + log_lik
        log_post -= np.max(log_post)
        unnorm = np.exp(log_post)
        post_probs = unnorm / unnorm.sum()

        return GridDistribution2d(
            x_grid=self._x_grid,
            y_grid=self._y_grid,
            probs=post_probs,
            x_name=self._x_name or 'α',
            y_name=self._y_name or 'β'
        )

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_contour(
            self,
            levels: int = 10,
            fill: bool = True,
            cmap: str = 'Blues',
            colorbar: bool = True,
            show_mode: bool = True,
            hdr: Optional[float] = None,
            title: Optional[str] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot 2D contour lines or filled contours of the joint distribution.

        :param levels: Number of contour levels or array of level values.
        :param fill: If True, renders filled contours (contourf), otherwise line contours.
        :param cmap: Colormap name.
        :param colorbar: Whether to display a colorbar.
        :param show_mode: Whether to mark the joint mode with a red marker.
        :param hdr: Optional float in (0, 1), e.g. 0.95, to draw the 95% HDR boundary line.
        :param title: Optional plot title.
        :param ax: Optional matplotlib Axes.
        """
        ax = ax or new_axes()
        xx, yy = self.meshgrid

        if fill:
            cs = ax.contourf(xx, yy, self._probs, levels=levels, cmap=cmap, **kwargs)
        else:
            cs = ax.contour(xx, yy, self._probs, levels=levels, cmap=cmap, **kwargs)

        if colorbar:
            plt.colorbar(cs, ax=ax)

        if hdr is not None:
            mask = self.hdr_mask(hdr).astype(float)
            ax.contour(xx, yy, mask, levels=[0.5], colors=['red'], linewidths=[2.0], linestyles=['--'])

        if show_mode:
            mx, my = self.mode()
            ax.plot(mx, my, marker='*', markersize=12, color='red', label=f'Mode ({mx:.2f}, {my:.2f})')
            ax.legend(loc='best')

        ax.set_xlabel(self._x_name)
        ax.set_ylabel(self._y_name)
        mu_x, mu_y = self.mean()
        if title is not None:
            ax.set_title(title)
        elif not ax.get_title():
            ax.set_title(
                f"2D Grid: {self._x_name} (μ={mu_x:.2f}) vs {self._y_name} (μ={mu_y:.2f}), r={self.corr():.2f}"
            )
        return ax

    def plot_heatmap(
            self,
            cmap: str = 'viridis',
            colorbar: bool = True,
            show_mode: bool = True,
            title: Optional[str] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot a 2D heatmap of the joint distribution.

        :param cmap: Colormap name.
        :param colorbar: Whether to display a colorbar.
        :param show_mode: Whether to mark the joint mode.
        :param title: Optional plot title.
        :param ax: Optional matplotlib Axes.
        """
        ax = ax or new_axes()
        # imshow extent: [left, right, bottom, top]
        extent = [self._x_grid[0], self._x_grid[-1], self._y_grid[0], self._y_grid[-1]]
        # Transpose probs so x corresponds to horizontal and y to vertical
        im = ax.imshow(
            self._probs.T,
            extent=extent,
            origin='lower',
            aspect='auto',
            cmap=cmap,
            **kwargs
        )
        if colorbar:
            plt.colorbar(im, ax=ax)

        if show_mode:
            mx, my = self.mode()
            ax.plot(mx, my, marker='*', markersize=12, color='red', label=f'Mode ({mx:.2f}, {my:.2f})')
            ax.legend(loc='best')

        ax.set_xlabel(self._x_name)
        ax.set_ylabel(self._y_name)
        mu_x, mu_y = self.mean()
        if title is not None:
            ax.set_title(title)
        elif not ax.get_title():
            ax.set_title(
                f"2D Grid Heatmap: {self._x_name} (μ={mu_x:.2f}), {self._y_name} (μ={mu_y:.2f})"
            )
        return ax

    def plot(
            self,
            kind: str = 'contour',
            **kwargs
    ) -> Axes:
        """
        Plot the 2D distribution.

        :param kind: 'contour' (default) or 'heatmap'.
        """
        if kind == 'contour':
            return self.plot_contour(**kwargs)
        elif kind == 'heatmap':
            return self.plot_heatmap(**kwargs)
        else:
            raise ValueError(f"kind must be 'contour' or 'heatmap', got '{kind}'.")

    def __str__(self) -> str:
        mu_x, mu_y = self.mean()
        return (
            f"GridDistribution2d(x_name='{self._x_name}', y_name='{self._y_name}', "
            f"shape={self.shape}, E[{self._x_name}]={mu_x:.3f}, E[{self._y_name}]={mu_y:.3f}, "
            f"corr={self.corr():.3f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


# Convenience Alias
BayesianGrid2d = GridDistribution2d
