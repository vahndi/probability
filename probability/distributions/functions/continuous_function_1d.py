from matplotlib.axes import Axes
from pandas import Series
from scipy.stats import rv_continuous
from typing import Iterable, overload, Optional

from probability.distributions.mixins.plottable_mixin import \
    ContinuousPlottableMixin
from mpl_format.axes.axis_utils import new_axes


class ContinuousFunction1d(object):

    def __init__(self,
                 distribution: rv_continuous,
                 method_name: str,
                 name: str,
                 parent: ContinuousPlottableMixin):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: ContinuousPlottableMixin = parent

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
        if isinstance(x, float) or isinstance(x, int):
            return self._method(x)
        elif isinstance(x, Iterable):
            return Series(index=x, data=self._method(x), name=self._name)

    def plot(self,
             x: Iterable,
             kind: str = 'line',
             color: str = 'C0',
             ax: Optional[Axes] = None,
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
        if self._method_name in ('pdf', 'cdf', 'logpdf'):
            if 'label' not in kwargs.keys():
                kwargs['label'] = self._parent.label
            data.plot(kind=kind, color=color, ax=ax, **kwargs)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))
        ax.set_xlabel(self._parent.x_label)

        if self._parent.y_label:
            ax.set_ylabel(self._parent.y_label)
        else:
            if self._method_name == 'pdf':
                ax.set_ylabel('P(X = x)')
            elif self._method_name == 'cdf':
                ax.set_ylabel('P(X ≤ x)')
            elif self._method_name == 'logpdf':
                ax.set_ylabel('log P(X = x)')
            else:
                ax.set_ylabel(self._name)

        return ax
