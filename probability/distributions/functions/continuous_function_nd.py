from itertools import product

from matplotlib.axes import Axes
from numpy import array, ndarray, mgrid, arange, dstack, meshgrid
from pandas import Series, MultiIndex
from scipy.stats import rv_continuous
from typing import overload, Iterable, Union

from probability.plots import new_axes


class ContinuousFunctionNd(object):

    def __init__(self, distribution: rv_continuous, method_name: str, name: str,
                 num_dims: int, parent: object):
        """
        :param distribution: The scipy distribution to calculate with.
        :param method_name: The name of the method to call on the distribution.
        :param name: An intuitive name for the function.
        :param num_dims: The number of dimensions, K, of the function.
        :param parent: The parent distribution object, used to call str(...) for series labels.
        """
        self._distribution = distribution
        self._num_dims = num_dims
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: object = parent

    @overload
    def at(self, x: Iterable[float]) -> float:
        pass

    @overload
    def at(self, x: Iterable[Iterable]) -> Series:
        pass

    @overload
    def at(self, x: ndarray) -> Series:
        pass

    def at(self, x):
        """
        Evaluate the function for each value of [x1, x2, ..., xk] given as x.

        :param x: [x1, x2, ..., xk] or [[x11, x12, ..., x1k], [x21, x22, ..., x2k], ...]
        """
        x = array(x)
        if x.ndim == 1:
            return self._method(x)
        elif x.ndim == 2:
            return Series(
                index=MultiIndex.from_arrays(
                    arrays=x.T,
                    names=[f'x{num}' for num in range(1, self._num_dims + 1)]
                ), data=self._method(x), name=f'{self._name}({self._parent})'
            )

    def plot(self, x1: Union[Iterable, ndarray], x2: Union[Iterable, ndarray],
             color_map: str = 'viridis', ax: Axes = None) -> Axes:
        """
        Plot the function.

        :param x1: Range of values of x1 to plot p(x1, x2) over.
        :param x2: Range of values of x2 to plot p(x1, x2) over.
        :param color_map: Optional colormap for the plot.
        :param ax: Optional matplotlib axes to plot on.
        """
        x1_grid, x2_grid = meshgrid(x1, x2)
        x1_x2 = dstack((x1_grid, x2_grid))
        f = self._method(x1_x2)
        ax = ax or new_axes()
        ax.contourf(x1_grid, x2_grid, f, cmap=color_map)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        return ax
