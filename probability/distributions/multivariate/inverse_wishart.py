from typing import Union, Optional, List

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import invwishart

from probability.custom_types.external_custom_types import FloatArray2d
from probability.distributions.continuous.inverse_gamma import InverseGamma


class InverseWishart(object):
    """
    Inverse-Wishart distribution for positive-definite covariance matrices.

    https://en.wikipedia.org/wiki/Inverse-Wishart_distribution

    Parameters:
        nu: Degrees of freedom (nu > d - 1).
        psi: Scale matrix (d x d positive definite matrix).
    """

    def __init__(
            self,
            nu: float,
            psi: FloatArray2d,
            names: Optional[List[str]] = None
    ):
        """
        Create a new Inverse-Wishart distribution.

        :param nu: Degrees of freedom (nu > d - 1).
        :param psi: Scale matrix (d x d).
        :param names: Optional variable names for dimensions.
        """
        if isinstance(psi, DataFrame):
            psi_df = psi
            if names is None:
                names = list(psi.columns)
        else:
            psi_arr = np.array(psi, dtype=float)
            d = psi_arr.shape[0]
            if names is None:
                names = [f'x{k}' for k in range(1, d + 1)]
            psi_df = DataFrame(data=psi_arr, index=names, columns=names)

        d = len(names)
        if nu <= d - 1:
            raise ValueError(f"Degrees of freedom (nu={nu}) must be greater than d - 1 = {d - 1}.")

        self._nu: float = float(nu)
        self._psi: DataFrame = psi_df
        self._names: List[str] = names
        self._dim: int = d
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution = invwishart(
            df=self._nu,
            scale=self._psi.to_numpy(dtype=float)
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
    def psi(self) -> DataFrame:
        """Scale matrix."""
        return self._psi

    @property
    def scale(self) -> DataFrame:
        """Alias for scale matrix."""
        return self._psi

    @property
    def dim(self) -> int:
        """Dimension d."""
        return self._dim

    @property
    def names(self) -> List[str]:
        """Variable names."""
        return self._names

    def mean(self) -> DataFrame:
        """
        Expected covariance matrix: Psi / (nu - d - 1) for nu > d + 1.
        """
        if self._nu > self._dim + 1:
            return self._psi / (self._nu - self._dim - 1.0)
        return DataFrame(np.nan, index=self._names, columns=self._names)

    def mode(self) -> DataFrame:
        """
        Mode covariance matrix: Psi / (nu + d + 1).
        """
        return self._psi / (self._nu + self._dim + 1.0)

    def rvs(
            self,
            num_samples: int = 1,
            random_state: Optional[int] = None
    ) -> Union[DataFrame, List[DataFrame]]:
        """
        Draw random covariance matrices from the distribution.

        :param num_samples: Number of sample matrices to draw.
        :param random_state: Optional random seed.
        :return: Single DataFrame (if num_samples == 1) or List of DataFrames.
        """
        samples = self._distribution.rvs(
            size=num_samples,
            random_state=random_state
        )
        if num_samples == 1:
            if samples.ndim == 3:
                samples = samples[0]
            return DataFrame(data=samples, index=self._names, columns=self._names)
        return [
            DataFrame(data=samples[i], index=self._names, columns=self._names)
            for i in range(num_samples)
        ]

    def marginal_variance(self, name_or_idx: Union[str, int]) -> InverseGamma:
        """
        Return the marginal Inverse-Gamma distribution for the diagonal variance element Sigma_jj:
            Sigma_jj ~ InverseGamma(alpha = (nu - d + 1) / 2, beta = Psi_jj / 2)
        """
        if isinstance(name_or_idx, int):
            idx = name_or_idx
        else:
            idx = self._names.index(name_or_idx)

        psi_jj = float(self._psi.iloc[idx, idx])
        alpha = (self._nu - self._dim + 1.0) / 2.0
        beta = psi_jj / 2.0
        return InverseGamma(alpha=alpha, beta=beta)

    def pdf(self, X: Union[ndarray, DataFrame]) -> float:
        """
        Evaluate probability density at covariance matrix X.
        """
        if isinstance(X, DataFrame):
            arr = X.to_numpy(dtype=float)
        else:
            arr = np.array(X, dtype=float)
        return float(self._distribution.pdf(arr))

    def __str__(self) -> str:
        return f'InverseWishart(ν={self._nu:.3f}, Ψ={self._psi.values.tolist()})'

    def __repr__(self) -> str:
        return f'InverseWishart(nu={self._nu}, psi={self._psi.values.tolist()})'
