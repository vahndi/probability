from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import Series

from probability.plotting import new_axes
from probability.utils import is_scalar


class GridDistribution1d(object):
    """
    1-Dimensional discrete grid approximation for Bayesian inference and distribution algebra.

    Represents a probability distribution over an ordered 1D support grid
    θ₁ < θ₂ < ... < θ_m with normalized probability masses p₁, p₂, ..., p_m.

    Supports:
      - Direct probability mass initialization and uniform presets
      - Initialization from any continuous or discrete distribution
      - Non-conjugate Bayesian sequential updating: posterior ∝ prior × likelihood
      - Exact summary metrics: mean, std, median, mode, quantiles, and HDIs
      - Distribution algebra: addition, subtraction, multiplication, and scaling
      - Inequality comparisons: P(X > Y), P(X < Y), P(X > c)
      - Plotting with shaded credible and highest density intervals (HDIs)
    """

    def __init__(
            self,
            grid: Sequence[float],
            probs: Optional[Sequence[float]] = None,
            name: Optional[str] = None
    ):
        """
        Create a new 1D Grid Distribution.

        :param grid: Sequence of grid points (numeric).
        :param probs: Optional sequence of probability masses. If None, defaults to uniform.
        :param name: Optional name for the parameter / variable.
        """
        grid_arr = np.asarray(grid, dtype=float)
        if grid_arr.ndim != 1 or len(grid_arr) < 2:
            raise ValueError("Grid must be a 1D sequence containing at least 2 points.")

        # Sort grid if not sorted
        sort_idx = np.argsort(grid_arr)
        grid_arr = grid_arr[sort_idx]

        if probs is None:
            probs_arr = np.full(len(grid_arr), 1.0 / len(grid_arr), dtype=float)
        else:
            probs_arr = np.asarray(probs, dtype=float)[sort_idx]
            if len(probs_arr) != len(grid_arr):
                raise ValueError(
                    f"Length of probs ({len(probs_arr)}) must match length of grid ({len(grid_arr)})."
                )
            if np.any(probs_arr < 0):
                raise ValueError("Probabilities must be non-negative.")
            total = probs_arr.sum()
            if total <= 0:
                raise ValueError("Sum of probabilities must be strictly positive.")
            probs_arr = probs_arr / total

        self._grid: ndarray = grid_arr
        self._probs: ndarray = probs_arr
        self._name: Optional[str] = name

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def grid(self) -> ndarray:
        """The 1D grid support points."""
        return self._grid

    @property
    def probs(self) -> ndarray:
        """The probability masses corresponding to each grid point."""
        return self._probs

    @property
    def name(self) -> Optional[str]:
        """Name of the distribution / parameter."""
        return self._name

    def __len__(self) -> int:
        return len(self._grid)

    # -------------------------------------------------------------------------
    # Alternative Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def uniform(
            cls,
            low: float,
            high: float,
            num_points: int = 101,
            name: Optional[str] = None
    ) -> 'GridDistribution1d':
        """
        Create a uniform grid distribution over [low, high].
        """
        if low >= high:
            raise ValueError(f"low ({low}) must be strictly less than high ({high}).")
        grid = np.linspace(low, high, num_points)
        return cls(grid=grid, name=name)

    @classmethod
    def from_distribution(
            cls,
            dist: Any,
            grid: Sequence[float],
            name: Optional[str] = None
    ) -> 'GridDistribution1d':
        """
        Initialize a grid distribution from any continuous or discrete distribution.

        :param dist: Distribution object (must support .pdf().at() or .pmf().at() or .pdf()).
        :param grid: Sequence of grid points.
        :param name: Optional name.
        """
        grid_arr = np.asarray(grid, dtype=float)
        if hasattr(dist, 'pdf'):
            pdf_fn = dist.pdf()
            if hasattr(pdf_fn, 'at'):
                probs = pdf_fn.at(grid_arr)
            else:
                probs = dist.pdf(grid_arr)
        elif hasattr(dist, 'pmf'):
            pmf_fn = dist.pmf()
            if hasattr(pmf_fn, 'at'):
                probs = pmf_fn.at(grid_arr)
            else:
                probs = dist.pmf(grid_arr)
        else:
            raise TypeError("dist must have a .pdf() or .pmf() method.")

        if hasattr(probs, 'to_numpy'):
            probs = probs.to_numpy(dtype=float)
        elif hasattr(probs, 'values'):
            probs = probs.values
        else:
            probs = np.asarray(probs, dtype=float)

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return cls(grid=grid_arr, probs=probs, name=name or getattr(dist, 'name', None))

    @classmethod
    def from_function(
            cls,
            fn: Callable[[float], float],
            grid: Sequence[float],
            name: Optional[str] = None
    ) -> 'GridDistribution1d':
        """
        Initialize a grid distribution from an unnormalized density function.
        """
        grid_arr = np.asarray(grid, dtype=float)
        try:
            probs = np.asarray(fn(grid_arr), dtype=float)
            if probs.shape != grid_arr.shape:
                probs = np.array([float(fn(x)) for x in grid_arr], dtype=float)
        except Exception:
            probs = np.array([float(fn(x)) for x in grid_arr], dtype=float)

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return cls(grid=grid_arr, probs=probs, name=name)

    # -------------------------------------------------------------------------
    # Bayesian Updating
    # -------------------------------------------------------------------------

    def update(
            self,
            likelihood_fn: Union[Callable[[float, Any], float], Callable[[ndarray, Any], ndarray], Sequence[float], ndarray],
            data: Any = None,
            inplace: bool = False
    ) -> 'GridDistribution1d':
        """
        Perform Bayesian updating given a likelihood function:
            posterior(θ) ∝ prior(θ) × likelihood(θ, data)

        :param likelihood_fn: Function mapping (theta, data) -> likelihood value(s),
                              or array of likelihood values matching grid shape.
        :param data: Observed data passed to the likelihood function.
        :param inplace: If True, mutate self; otherwise return a new GridDistribution1d.
        :return: Updated GridDistribution1d.
        """
        if not callable(likelihood_fn):
            return self.update_likelihood(likelihood=likelihood_fn, inplace=inplace)

        try:
            # Try vectorized call
            likes = np.asarray(likelihood_fn(self._grid, data), dtype=float)
            if likes.shape != self._grid.shape:
                likes = np.array([float(likelihood_fn(theta, data)) for theta in self._grid], dtype=float)
        except Exception:
            likes = np.array([float(likelihood_fn(theta, data)) for theta in self._grid], dtype=float)

        likes = np.nan_to_num(likes, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(likes < 0):
            raise ValueError("Likelihood values must be non-negative.")

        unnorm_post = self._probs * likes
        total = unnorm_post.sum()
        if total <= 0:
            raise ValueError("Likelihood resulted in zero posterior probability across the entire grid support.")

        norm_post = unnorm_post / total
        if inplace:
            self._probs = norm_post
            return self
        return GridDistribution1d(grid=self._grid, probs=norm_post, name=self._name)

    def update_likelihood(
            self,
            likelihood: Union[ndarray, Sequence[float]],
            inplace: bool = False
    ) -> 'GridDistribution1d':
        """
        Perform Bayesian updating given a direct array of likelihood values across the grid:
            posterior(θ) ∝ prior(θ) × likelihood(θ)

        :param likelihood: Array or sequence of likelihood values matching the grid length.
        :param inplace: If True, mutate self; otherwise return a new GridDistribution1d.
        :return: Updated GridDistribution1d.
        """
        likes = np.asarray(likelihood, dtype=float)
        if likes.shape != self._grid.shape:
            raise ValueError(f"Likelihood array shape {likes.shape} must match grid shape {self._grid.shape}.")
        likes = np.nan_to_num(likes, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(likes < 0):
            raise ValueError("Likelihood values must be non-negative.")

        unnorm_post = self._probs * likes
        total = unnorm_post.sum()
        if total <= 0:
            raise ValueError("Likelihood resulted in zero posterior probability across the entire grid support.")

        norm_post = unnorm_post / total
        if inplace:
            self._probs = norm_post
            return self
        return GridDistribution1d(grid=self._grid, probs=norm_post, name=self._name)

    def update_binomial(
            self,
            k: int,
            n: int,
            inplace: bool = False
    ) -> 'GridDistribution1d':
        """
        Bayesian update for Binomial data (k successes in n trials).
        Grid θ represents the success probability p in [0, 1].
        """
        clipped_grid = np.clip(self._grid, 0.0, 1.0)
        likes = (clipped_grid ** k) * ((1.0 - clipped_grid) ** (n - k))
        return self.update(lambda theta, data: likes, inplace=inplace)

    def update_poisson(
            self,
            k: Union[int, Sequence[int]],
            inplace: bool = False
    ) -> 'GridDistribution1d':
        """
        Bayesian update for Poisson count data.
        Grid θ represents the rate parameter λ > 0.
        """
        k_arr = np.atleast_1d(k)
        sum_k = float(k_arr.sum())
        num_obs = len(k_arr)
        pos_grid = np.maximum(self._grid, 1e-12)
        # Likelihood proportional to theta^(sum k) * exp(-num_obs * theta)
        # Use log-scale for numerical stability
        log_likes = sum_k * np.log(pos_grid) - num_obs * pos_grid
        likes = np.exp(log_likes - np.max(log_likes))
        return self.update(lambda theta, data: likes, inplace=inplace)

    def update_normal(
            self,
            data: Union[float, Sequence[float]],
            sigma: float,
            inplace: bool = False
    ) -> 'GridDistribution1d':
        """
        Bayesian update for Normal data with known standard deviation σ.
        Grid θ represents the unknown mean μ.
        """
        x_arr = np.atleast_1d(data)
        n = len(x_arr)
        x_bar = float(x_arr.mean())
        # Likelihood proportional to exp(-n * (theta - x_bar)^2 / (2 * sigma^2))
        likes = np.exp(-n * ((self._grid - x_bar) ** 2) / (2.0 * sigma ** 2))
        return self.update(lambda theta, d: likes, inplace=inplace)

    # -------------------------------------------------------------------------
    # Summary Statistics & Metrics
    # -------------------------------------------------------------------------

    def mean(self) -> float:
        """Expected value: E[θ] = sum(θ_i * p_i)."""
        return float(np.sum(self._grid * self._probs))

    def var(self) -> float:
        """Variance: Var(θ) = sum((θ_i - E[θ])² * p_i)."""
        mu = self.mean()
        return float(np.sum(((self._grid - mu) ** 2) * self._probs))

    def std(self) -> float:
        """Standard deviation: sqrt(Var(θ))."""
        return float(np.sqrt(self.var()))

    def mode(self) -> float:
        """Point on the grid with maximum probability mass."""
        idx = int(np.argmax(self._probs))
        return float(self._grid[idx])

    def median(self) -> float:
        """Median (50th percentile grid point)."""
        cum = np.cumsum(self._probs)
        idx = int(np.searchsorted(cum, 0.50))
        return float(self._grid[min(idx, len(self._grid) - 1)])

    def cdf(self) -> Series:
        """Cumulative distribution function as a pandas Series."""
        return Series(data=np.cumsum(self._probs), index=self._grid, name='cdf')

    def pmf(self) -> Series:
        """Probability mass function as a pandas Series."""
        return Series(data=self._probs, index=self._grid, name='pmf')

    def at(self, x: float) -> float:
        """
        Evaluate probability mass / density at value x (exact or interpolated).
        """
        return float(np.interp(x, self._grid, self._probs, left=0.0, right=0.0))

    def quantile(self, q: float, interpolate: bool = True) -> float:
        """
        Return the value at quantile q in [0, 1].

        :param q: Quantile in [0, 1].
        :param interpolate: If True (default), use linear interpolation between grid points;
                            if False, return the step grid point where CDF >= q.
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"Quantile q ({q}) must be in [0, 1].")
        cum = np.cumsum(self._probs)
        if interpolate:
            return float(np.interp(q, cum, self._grid))
        idx = int(np.searchsorted(cum, q))
        return float(self._grid[min(idx, len(self._grid) - 1)])

    def credible_interval(self, prob: float = 0.95) -> Tuple[float, float]:
        """
        Return the equal-tailed credible interval containing probability mass `prob`.
        """
        if not (0.0 < prob < 1.0):
            raise ValueError("prob must be strictly between 0 and 1.")
        alpha = 1.0 - prob
        low = self.quantile(alpha / 2.0)
        high = self.quantile(1.0 - alpha / 2.0)
        return (low, high)

    def interval(self, prob: float = 0.95) -> Tuple[float, float]:
        """Alias for credible_interval."""
        return self.credible_interval(prob)

    def hdi(self, prob: float = 0.95) -> Tuple[float, float]:
        """
        Return the Highest Density Interval (HDI) containing at least `prob` probability mass.
        Finds the shortest contiguous interval [θ_i, θ_j] such that sum_{k=i}^j p_k >= prob.
        """
        if not (0.0 < prob < 1.0):
            raise ValueError("prob must be strictly between 0 and 1.")
        cumsum = np.cumsum(self._probs)
        min_width = float('inf')
        best_interval = (float(self._grid[0]), float(self._grid[-1]))

        for i in range(len(self._grid)):
            target = (cumsum[i - 1] if i > 0 else 0.0) + prob
            j_idx = np.searchsorted(cumsum, target)
            if j_idx < len(self._grid):
                width = self._grid[j_idx] - self._grid[i]
                if width < min_width:
                    min_width = width
                    best_interval = (float(self._grid[i]), float(self._grid[j_idx]))

        return best_interval

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
            self,
            num_samples: int = 1,
            random_state: Optional[int] = None
    ) -> Union[float, ndarray]:
        """
        Draw samples with replacement from the grid according to probability masses.
        """
        if random_state is not None:
            np.random.seed(random_state)
        samples = np.random.choice(self._grid, size=num_samples, p=self._probs)
        if num_samples == 1:
            return float(samples[0])
        return samples

    def rvs(
            self,
            num_samples: int = 1,
            random_state: Optional[int] = None
    ) -> Series:
        """
        Sample `num_samples` values from the distribution as a pandas Series.
        """
        draws = self.sample(num_samples=num_samples, random_state=random_state)
        return Series(np.atleast_1d(draws), name=self._name)

    # -------------------------------------------------------------------------
    # Comparisons: Return float probabilities in [0, 1]
    # -------------------------------------------------------------------------

    def __gt__(self, other: Any) -> float:
        if is_scalar(other):
            mask = self._grid > float(other)
            return float(self._probs[mask].sum())
        elif isinstance(other, GridDistribution1d):
            outer_p = np.outer(self._probs, other._probs)
            gt_mask = np.greater.outer(self._grid, other._grid)
            return float(outer_p[gt_mask].sum())
        elif hasattr(other, 'rvs'):
            # Monte Carlo fallback for arbitrary distributions
            n = 100_000
            return float((self.rvs(n).values > other.rvs(n)).mean())
        raise TypeError(f"Cannot compare GridDistribution1d with {type(other)}.")

    def __ge__(self, other: Any) -> float:
        if is_scalar(other):
            mask = self._grid >= float(other)
            return float(self._probs[mask].sum())
        elif isinstance(other, GridDistribution1d):
            outer_p = np.outer(self._probs, other._probs)
            ge_mask = np.greater_equal.outer(self._grid, other._grid)
            return float(outer_p[ge_mask].sum())
        elif hasattr(other, 'rvs'):
            n = 100_000
            return float((self.rvs(n).values >= other.rvs(n)).mean())
        raise TypeError(f"Cannot compare GridDistribution1d with {type(other)}.")

    def __lt__(self, other: Any) -> float:
        if is_scalar(other):
            mask = self._grid < float(other)
            return float(self._probs[mask].sum())
        elif isinstance(other, GridDistribution1d):
            outer_p = np.outer(self._probs, other._probs)
            lt_mask = np.less.outer(self._grid, other._grid)
            return float(outer_p[lt_mask].sum())
        elif hasattr(other, 'rvs'):
            n = 100_000
            return float((self.rvs(n).values < other.rvs(n)).mean())
        raise TypeError(f"Cannot compare GridDistribution1d with {type(other)}.")

    def __le__(self, other: Any) -> float:
        if is_scalar(other):
            mask = self._grid <= float(other)
            return float(self._probs[mask].sum())
        elif isinstance(other, GridDistribution1d):
            outer_p = np.outer(self._probs, other._probs)
            le_mask = np.less_equal.outer(self._grid, other._grid)
            return float(outer_p[le_mask].sum())
        elif hasattr(other, 'rvs'):
            n = 100_000
            return float((self.rvs(n).values <= other.rvs(n)).mean())
        raise TypeError(f"Cannot compare GridDistribution1d with {type(other)}.")

    def __eq__(self, other: Any) -> float:
        if is_scalar(other):
            idx = np.where(np.isclose(self._grid, float(other)))[0]
            if len(idx) > 0:
                return float(self._probs[idx].sum())
            return 0.0
        elif isinstance(other, GridDistribution1d):
            outer_p = np.outer(self._probs, other._probs)
            eq_mask = np.isclose(np.subtract.outer(self._grid, other._grid), 0.0)
            return float(outer_p[eq_mask].sum())
        return 0.0

    # -------------------------------------------------------------------------
    # Distribution Arithmetic
    # -------------------------------------------------------------------------

    def __neg__(self) -> 'GridDistribution1d':
        new_grid = -self._grid[::-1]
        new_probs = self._probs[::-1]
        return GridDistribution1d(grid=new_grid, probs=new_probs, name=f'-{self._name}' if self._name else None)

    def _combine_with_other(
            self,
            other: 'GridDistribution1d',
            op: Callable[[ndarray, ndarray], ndarray],
            n_bins: int = 150
    ) -> 'GridDistribution1d':
        """
        Combine two independent GridDistribution1d objects under a binary operation.
        """
        z_outer = op(self._grid[:, None], other._grid[None, :]).ravel()
        w_outer = np.outer(self._probs, other._probs).ravel()

        z_min, z_max = z_outer.min(), z_outer.max()
        if np.isclose(z_min, z_max):
            return GridDistribution1d(grid=[z_min - 1e-6, z_min + 1e-6], probs=[0.5, 0.5])

        # Bin onto a smooth grid
        bins = np.linspace(z_min, z_max, n_bins + 1)
        hist, bin_edges = np.histogram(z_outer, bins=bins, weights=w_outer)
        grid_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return GridDistribution1d(grid=grid_centers, probs=hist)

    def __add__(self, other: Any) -> 'GridDistribution1d':
        if is_scalar(other):
            c = float(other)
            return GridDistribution1d(grid=self._grid + c, probs=self._probs.copy(), name=self._name)
        elif isinstance(other, GridDistribution1d):
            return self._combine_with_other(other, np.add)
        raise TypeError(f"Unsupported operand type for +: GridDistribution1d and {type(other)}")

    def __radd__(self, other: Any) -> 'GridDistribution1d':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'GridDistribution1d':
        if is_scalar(other):
            c = float(other)
            return GridDistribution1d(grid=self._grid - c, probs=self._probs.copy(), name=self._name)
        elif isinstance(other, GridDistribution1d):
            return self._combine_with_other(other, np.subtract)
        raise TypeError(f"Unsupported operand type for -: GridDistribution1d and {type(other)}")

    def __rsub__(self, other: Any) -> 'GridDistribution1d':
        if is_scalar(other):
            return (-self).__add__(other)
        raise TypeError(f"Unsupported operand type for -: {type(other)} and GridDistribution1d")

    def __mul__(self, other: Any) -> 'GridDistribution1d':
        if is_scalar(other):
            c = float(other)
            if c > 0:
                return GridDistribution1d(grid=self._grid * c, probs=self._probs.copy(), name=self._name)
            elif c < 0:
                return GridDistribution1d(grid=(self._grid * c)[::-1], probs=self._probs[::-1].copy(), name=self._name)
            else:
                return GridDistribution1d(grid=[-1e-6, 1e-6], probs=[0.5, 0.5], name=self._name)
        elif isinstance(other, GridDistribution1d):
            return self._combine_with_other(other, np.multiply)
        raise TypeError(f"Unsupported operand type for *: GridDistribution1d and {type(other)}")

    def __rmul__(self, other: Any) -> 'GridDistribution1d':
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'GridDistribution1d':
        if is_scalar(other):
            c = float(other)
            if c == 0:
                raise ZeroDivisionError("division by zero")
            return self.__mul__(1.0 / c)
        elif isinstance(other, GridDistribution1d):
            return self._combine_with_other(other, np.divide)
        raise TypeError(f"Unsupported operand type for /: GridDistribution1d and {type(other)}")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot(
            self,
            kind: str = 'line',
            ci: Optional[float] = None,
            hdi: Optional[float] = None,
            fill_alpha: float = 0.3,
            color: str = 'tab:blue',
            label: Optional[str] = None,
            title: Optional[str] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot the probability mass or density curve over the 1D grid.

        :param kind: 'line' (default) or 'bar'.
        :param ci: Optional equal-tailed credible interval to shade (e.g. 0.95).
        :param hdi: Optional highest density interval to shade (e.g. 0.95).
        :param fill_alpha: Transparency for shaded interval regions.
        :param color: Plotting color.
        :param label: Legend label.
        :param title: Optional plot title.
        :param ax: Optional matplotlib Axes.
        """
        ax = ax or new_axes()
        lbl = label or self._name or 'Grid PMF'
        if 'title' in kwargs and title is None:
            title = kwargs.pop('title')

        if kind == 'line':
            ax.plot(self._grid, self._probs, color=color, label=lbl, **kwargs)
        elif kind == 'bar':
            width = np.diff(self._grid).min() * 0.9 if len(self._grid) > 1 else 1.0
            ax.bar(self._grid, self._probs, width=width, color=color, label=lbl, **kwargs)
        else:
            raise ValueError(f"kind must be 'line' or 'bar', got '{kind}'.")

        # Shade interval
        if ci is not None:
            low, high = self.credible_interval(ci)
            mask = (self._grid >= low) & (self._grid <= high)
            ax.fill_between(
                self._grid[mask], 0, self._probs[mask],
                color=color, alpha=fill_alpha,
                label=f'{ci:.0%} CI [{low:.2f}, {high:.2f}]'
            )

        if hdi is not None:
            low, high = self.hdi(hdi)
            mask = (self._grid >= low) & (self._grid <= high)
            ax.fill_between(
                self._grid[mask], 0, self._probs[mask],
                color=color, alpha=fill_alpha,
                label=f'{hdi:.0%} HDI [{low:.2f}, {high:.2f}]'
            )

        ax.set_xlabel(self._name or 'θ')
        ax.set_ylabel('Probability')
        if title is not None:
            ax.set_title(title)
        elif not ax.get_title():
            ax.set_title(f"Grid Distribution (mean={self.mean():.3f}, std={self.std():.3f})")
        ax.legend(loc='best')
        return ax

    def __str__(self) -> str:
        return (
            f"GridDistribution1d(n_points={len(self._grid)}, "
            f"range=[{self._grid[0]:.3f}, {self._grid[-1]:.3f}], "
            f"mean={self.mean():.3f}, std={self.std():.3f})"
        )

    def __repr__(self) -> str:
        return (
            f"GridDistribution1d(grid=[{self._grid[0]:.3f}, ..., {self._grid[-1]:.3f}], "
            f"mean={self.mean():.3f}, std={self.std():.3f})"
        )


BayesianGrid1d = GridDistribution1d
