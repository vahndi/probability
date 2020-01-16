from matplotlib.axes import Axes
from pandas import Series
from scipy.stats import rv_discrete
from typing import Iterable, overload

from probability.distributions.mixins.plottable_mixin import PlottableMixin
from probability.plots import new_axes


class DiscreteFunction1d(object):

    def __init__(self, distribution: rv_discrete, method_name: str, name: str,
                 parent: PlottableMixin):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: PlottableMixin = parent

    @overload
    def at(self, k: int) -> int:
        pass

    @overload
    def at(self, k: Iterable[int]) -> Series:
        pass

    def at(self, k):
        """
        Evaluation of the function for each value of k.
        """
        if isinstance(k, int):
            return self._method(k)
        elif isinstance(k, Iterable):
            return Series(index=k, data=self._method(k), name=self._name)

    def plot(self, k: Iterable[int], color: str = 'C0', kind: str = 'bar', ax: Axes = None,
             **kwargs) -> Axes:
        """
        Plot the function.

        :param k: Range of values of k to plot p(k) over.
        :param color: Optional color for the series.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        data: Series = self.at(k)
        ax = ax or new_axes()

        # special kwargs
        vlines = None
        if 'vlines' in kwargs.keys():
            vlines = kwargs.pop('vlines')

        if self._name == 'PMF':
            data.plot(kind=kind, label=self._parent.label, color=color,
                      ax=ax, **kwargs)
        elif self._name == 'CDF':
            data.plot(kind='line', label=self._parent.label, color=color,
                      drawstyle='steps-post', ax=ax,
                      **kwargs)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))
        if vlines:
            ax.vlines(x=k, ymin=0, ymax=data.values, color=color)
        ax.set_xlabel(self._parent.x_label)
        ax.set_ylabel(self._name)
        return ax
