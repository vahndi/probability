from typing import Union, Iterable

from mpl_toolkits.axes_grid1.mpl_axes import Axes
from numpy import ndarray
from pandas import Series, DataFrame
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multi_rv_generic

from probability.custom_types.external_custom_types import FloatArray1d, \
    FloatArray2d
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.dimension_mixins import NdMixin
from probability.distributions.mixins.rv_mixins import \
    RVSNdMixin, EntropyMixin, PDFNdMixin, CDFContinuousNdMixin


class MVNormal(
    NdMixin,
    RVSNdMixin,
    CDFContinuousNdMixin,
    PDFNdMixin,
    EntropyMixin,
    CalculableMixin,
    object
):
    """
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    def __init__(self,
                 mu: Union[FloatArray1d, dict, Series],
                 sigma: FloatArray2d):
        """
        Create a new multi-variate normal distribution.

        :param mu: List or mapping of means.
        :param sigma: List or mapping of standard deviations.
        """
        if (
                (isinstance(mu, dict) or isinstance(mu, Series)) and
                (isinstance(sigma, dict) or isinstance(sigma, Series))
        ):
            if not set(mu.keys()) == set(sigma.keys()):
                raise ValueError('keys for mu and sigma must be identical')

        if isinstance(mu, dict) or isinstance(mu, Series):
            names = mu.keys()
        else:
            names = [f'x{k}' for k in range(1, len(mu) + 1)]

        if isinstance(mu, dict):
            mu = Series(mu)
        elif not isinstance(mu, Series):
            mu = Series(data=mu, index=names)

        self._mu: Series = mu
        self._sigma: DataFrame = DataFrame(
            data=sigma, index=names, columns=names
        )
        self._set_names(names)
        self._num_dims = len(self._mu)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: multi_rv_generic = multivariate_normal(
            mean=self._mu.values,
            cov=self._sigma.values
        )

    @property
    def mu(self) -> Series:
        return self._mu

    @mu.setter
    def mu(self, value: Series):
        self._mu = value

    @property
    def sigma(self) -> DataFrame:
        return self._sigma

    @sigma.setter
    def sigma(self, value: DataFrame):
        self._sigma = value

    def plot_2d(self,
                x1: Union[Iterable, ndarray], x2: Union[Iterable, ndarray],
                color_map: str = 'viridis', ax: Axes = None,
                **kwargs) -> Axes:
        """
        Plot the function.

        :param x1: Range of values of x1 to plot p(x1, x2) over.
        :param x2: Range of values of x2 to plot p(x1, x2) over.
        :param color_map: Optional colormap for the plot.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments to pass to plot method.
        """
        return self.pdf().plot_2d(
            x1=x1, x2=x2, color_map=color_map, ax=ax, **kwargs
        )

    def __str__(self):

        return f'MVNormal(μ={self._mu}, Σ={self._sigma.values})'

    def __repr__(self):

        return f'MVNormal(mu={self._mu}, sigma={self._sigma.values})'
