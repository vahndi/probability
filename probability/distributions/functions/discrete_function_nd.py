from typing import overload, Iterable, Union

from matplotlib.axes import Axes
from numpy import array, ndarray
from pandas import Series, MultiIndex
from scipy.stats import rv_discrete

from probability.plots import new_axes


class DiscreteFunctionNd(object):

    def __init__(self, distribution: rv_discrete, method_name: str, name: str,
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
    def at(self, x: Iterable[int]) -> float:
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
                ), data=self._method(x)
            )

    def plot(self, x: Union[Iterable[Iterable], ndarray],
             color: str = 'C0', ax: Axes = None) -> Axes:
        """
        Plot the function.

        :param x: Range of values of x to plot p(x) over.
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes to plot on.
        """
        x = array(x)
        data = self.at(x)
        if x.ndim != 2:
            raise ValueError('x must have 2 dimensions: [num_points, K]')
        ax = ax or new_axes()
        if self._num_dims > 2:
            data = data.sort_values(ascending=False)
        data.plot.bar(color=color, label=str(self._parent), ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel(self._name)
        ax.legend(loc='upper right')
        return ax
