from math import sqrt
from typing import Union, Iterable, Optional, List

from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
from pandas import Series, DataFrame
from scipy.stats import multivariate_t

from probability.custom_types.external_custom_types import FloatArray1d, FloatArray2d
from probability.distributions.continuous.students_t import StudentsT
from probability.distributions.mixins.dimension_mixins import NdMixin
from probability.distributions.mixins.rv_mixins import RVSNdMixin, PDFNdMixin


class MVStudentsT(
    NdMixin,
    RVSNdMixin,
    PDFNdMixin,
    object
):
    """
    Multivariate Student's t distribution.

    https://en.wikipedia.org/wiki/Multivariate_t-distribution
    """

    def __init__(
            self,
            nu: float,
            mu: Union[FloatArray1d, dict, Series],
            sigma: FloatArray2d,
            names: Optional[List[str]] = None
    ):
        """
        Create a new multivariate Student's t distribution.

        :param nu: Degrees of freedom (nu > 0).
        :param mu: Location vector (mean when nu > 1).
        :param sigma: Scale matrix (positive definite d x d).
        :param names: Optional variable names for dimensions.
        """
        if nu <= 0:
            raise ValueError("Degrees of freedom (nu) must be positive.")

        if isinstance(mu, (dict, Series)):
            if names is None:
                names = list(mu.keys())
        elif names is None:
            names = [f'x{k}' for k in range(1, len(mu) + 1)]

        if isinstance(mu, dict):
            mu_series = Series(mu)
        elif isinstance(mu, Series):
            mu_series = mu
        else:
            mu_series = Series(data=mu, index=names)

        if isinstance(sigma, DataFrame):
            sigma_df = sigma
        else:
            sigma_df = DataFrame(data=sigma, index=names, columns=names)

        self._nu: float = float(nu)
        self._mu: Series = mu_series
        self._sigma: DataFrame = sigma_df
        self._set_names(names)
        self._num_dims = len(self._mu)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution = multivariate_t(
            df=self._nu,
            loc=self._mu.to_numpy(dtype=float),
            shape=self._sigma.to_numpy(dtype=float)
        )

    @property
    def nu(self) -> float:
        """Degrees of freedom."""
        return self._nu

    @property
    def df(self) -> float:
        """Alias for degrees of freedom."""
        return self._nu

    @property
    def mu(self) -> Series:
        """Location vector."""
        return self._mu

    @property
    def loc(self) -> Series:
        """Alias for location vector."""
        return self._mu

    @property
    def sigma(self) -> DataFrame:
        """Scale matrix."""
        return self._sigma

    @property
    def shape(self) -> DataFrame:
        """Alias for scale matrix."""
        return self._sigma

    @property
    def num_dims(self) -> int:
        """Number of dimensions."""
        return self._num_dims

    def mean(self) -> Series:
        """
        Mean vector (defined only for nu > 1).
        """
        if self._nu > 1.0:
            return self._mu.copy()
        return Series(np.nan, index=self._names)

    def cov(self) -> DataFrame:
        """
        Covariance matrix: Sigma * (nu / (nu - 2)) (defined only for nu > 2).
        """
        if self._nu > 2.0:
            factor = self._nu / (self._nu - 2.0)
            return self._sigma * factor
        return DataFrame(np.nan, index=self._names, columns=self._names)

    def marginal(self, name_or_idx: Union[str, int]) -> StudentsT:
        """
        Return the 1D marginal Student's t distribution for a single component.

        For x ~ t_nu(mu, Sigma), the j-th component marginally follows
        x_j ~ StudentsT(nu=nu, mu=mu_j, sigma=sqrt(Sigma_jj)).
        """
        if isinstance(name_or_idx, int):
            idx = name_or_idx
        else:
            idx = self._names.index(name_or_idx)

        mu_j = float(self._mu.iloc[idx])
        sigma_jj = float(self._sigma.iloc[idx, idx])
        return StudentsT(
            nu=self._nu,
            mu=mu_j,
            sigma=sqrt(sigma_jj)
        )

    def plot_2d(
            self,
            x1: Union[Iterable, ndarray],
            x2: Union[Iterable, ndarray],
            color_map: str = 'viridis',
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot 2D density contours or heatmap.
        """
        return self.pdf().plot_2d(
            x1=x1, x2=x2, color_map=color_map, ax=ax, **kwargs
        )

    def __str__(self) -> str:
        return f'MVStudentsT(ν={self._nu:.3f}, μ={self._mu.to_dict()}, Σ={self._sigma.values.tolist()})'

    def __repr__(self) -> str:
        return f'MVStudentsT(nu={self._nu}, mu={self._mu.to_dict()}, sigma={self._sigma.values.tolist()})'
