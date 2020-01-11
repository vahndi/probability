from typing import Union, Iterable

from mpl_toolkits.axes_grid1.mpl_axes import Axes
from numpy import array, ndarray
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multi_rv_generic

from probability.custom_types import FloatOrFloatArray1d, FloatOrFloatArray2d
from probability.distributions.mixins.rv_mixins import RVSNdMixin, EntropyMixin, PDFNdMixin, CDFContinuousNdMixin


class MVNormal(
    RVSNdMixin, CDFContinuousNdMixin, PDFNdMixin, EntropyMixin,
    object
):

    def __init__(self, mu: FloatOrFloatArray1d, sigma: FloatOrFloatArray2d):

        self._mu: ndarray = array(mu)
        if isinstance(mu, float):
            self._num_dims = 1
        else:
            self._num_dims = len(self._mu)
        self._sigma: ndarray = array(sigma)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: multi_rv_generic = multivariate_normal(mean=self._mu, cov=self._sigma)

    @property
    def mu(self) -> ndarray:
        return self._mu

    @mu.setter
    def mu(self, value: FloatOrFloatArray1d):
        self._mu = value

    @property
    def sigma(self) -> ndarray:
        return self._sigma

    @sigma.setter
    def sigma(self, value: FloatOrFloatArray2d):
        self._sigma = value

    def plot_2d(self, x1: Union[Iterable, ndarray], x2: Union[Iterable, ndarray],
                color_map: str = 'viridis', ax: Axes = None) -> Axes:
        """
        Plot the function.

        :param x1: Range of values of x1 to plot p(x1, x2) over.
        :param x2: Range of values of x2 to plot p(x1, x2) over.
        :param color_map: Optional colormap for the plot.
        :param ax: Optional matplotlib axes to plot on.
        """
        return self.pdf().plot_2d(x1=x1, x2=x2, color_map=color_map, ax=ax)

    def __str__(self):

        return f'MVNormal(μ={self._mu}, Σ={self._sigma.tolist()})'

    def __repr__(self):

        return f'MVNormal(mu={self._mu}, sigma={self._sigma.tolist()})'
