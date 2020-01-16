from matplotlib.axes import Axes
from pandas import Series
from scipy.stats import rv_continuous
from typing import Iterable, overload

from probability.distributions.mixins.plottable_mixin import PlottableMixin
from probability.plots import new_axes


class ContinuousFunction1d(object):

    def __init__(self, distribution: rv_continuous, method_name: str, name: str,
                 parent: PlottableMixin):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: PlottableMixin = parent

    @overload
    def at(self, x: float) -> float:
        pass

    @overload
    def at(self, x: Iterable) -> Series:
        pass

    def at(self, x):
        """
        Log of the probability density function of the given RV.
        """
        if isinstance(x, float):
            return self._method(x)
        elif isinstance(x, Iterable):
            return Series(index=x, data=self._method(x), name=self._name)

    def plot(self, x: Iterable, kind: str = 'line', color: str = 'C0', ax: Axes = None,
             **kwargs) -> Axes:
        """
        Plot the function.

        :param x: Range of values of x to plot p(x) over.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        data: Series = self.at(x)
        ax = ax or new_axes()
        if self._name in ('PDF', 'CDF', 'log(PDF)'):
            data.plot(kind=kind, label=self._parent.label, color=color, ax=ax, **kwargs)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))
        ax.set_xlabel(self._parent.x_label)
        ax.set_ylabel(self._name)
        return ax
