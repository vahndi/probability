from typing import overload, Iterable, Union, Optional

from matplotlib.axes import Axes
from mpl_format.axes.axis_utils import new_axes
from numpy import array, ndarray
from pandas import Series, MultiIndex, concat
from scipy.stats import rv_discrete
from seaborn import lineplot


class DiscreteFunctionNd(object):

    def __init__(self, distribution: rv_discrete, method_name: str, name: str,
                 num_dims: int, parent: object):
        """
        :param distribution: The scipy distribution to calculate with.
        :param method_name: The name of the method to call on the distribution.
        :param name: An intuitive name for the function.
        :param num_dims: The number of dimensions, K, of the function.
        :param parent: The parent distribution object, used to call str(...)
                       for series labels.
        """
        self._distribution = distribution
        self._num_dims = num_dims
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: object = parent

    @overload
    def at(self, k: Iterable[int]) -> float:
        pass

    @overload
    def at(self, k: Iterable[Iterable]) -> Series:
        pass

    @overload
    def at(self, k: ndarray) -> Series:
        pass

    def at(self, k):
        """
        Evaluate the function for each value of [x1, x2, ..., xk] given as x.

        :param k: [x1, x2, ..., xk] or [[x11, x12, ..., x1k],
                                        [x21, x22, ..., x2k],
                                        ...]
        """
        k = array(k)
        if k.ndim == 1:
            return self._method(k)
        elif k.ndim == 2:
            return Series(
                index=MultiIndex.from_arrays(
                    arrays=k.T,
                    names=[f'x{num}' for num in range(1, self._num_dims + 1)]
                ), data=self._method(k)
            )

    def plot(self, k: Union[Iterable[Iterable], ndarray],
             ax: Optional[Axes] = None,
             **kwargs) -> Axes:
        """
        Plot the most probable values of the function.

        :param k: Range of values of x to plot p(x) over.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for bar plot.
        """
        k = array(k)
        if k.ndim != 2:
            raise ValueError('k must have 2 dimensions: [num_points, K]')
        data = self.at(k)
        # filter to "top" most likely
        data = data.rename('p').reset_index()
        new_data = []
        for name in self._parent.names:
            new_frame = data[[name, 'p']].rename(columns={name: 'x'})
            new_frame['$x_k$'] = name
            new_data.append(new_frame)
        data = concat(new_data)
        ax = ax or new_axes()
        lineplot(data=data, x='x', y='p', hue='$x_k$', ax=ax, **kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel(f'{self._name}(X)')
        ax.legend(loc='upper right')
        return ax

    def plot_top(self, top: int,
                 ax: Optional[Axes] = None,
                 **kwargs) -> Axes:
        """
        Plot the probability of the most likely values of the distribution.

        :param top: The number of top values to plot probability for.
        :param ax: Axes to plot on.
        :param kwargs: Additional arguments to pass to bar plot method.
        """
        k = array(self._parent.permutations())
        if k.ndim != 2:
            raise ValueError('k must have 2 dimensions: [num_points, K]')
        data = self.at(k)
        data = data.sort_values(ascending=False).head(top)
        ax = ax or new_axes()
        data.plot.bar(label=str(self._parent), ax=ax, **kwargs)
        ax.set_xlabel('K')
        ax.set_ylabel(f'{self._name}(K)')
        ax.legend(loc='upper right')
        return ax
