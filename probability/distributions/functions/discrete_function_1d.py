from matplotlib.axes import Axes
from pandas import Series
from scipy.stats import rv_discrete
from typing import Iterable, overload, Optional

from probability.distributions.mixins.plottable_mixin import \
    DiscretePlottableMixin
from mpl_format.axes.axis_utils import new_axes


class DiscreteFunction1d(object):

    def __init__(self,
                 distribution: rv_discrete,
                 method_name: str,
                 name: str,
                 parent: DiscretePlottableMixin):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: DiscretePlottableMixin = parent

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

    def plot(self, k: Iterable[int],
             color: str = 'C0',
             kind: str = 'bar',
             ax: Optional[Axes] = None,
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
        if 'label' not in kwargs.keys():
            kwargs['label'] = self._parent.label

        if self._method_name == 'pmf':
            data.plot(kind=kind, color=color, ax=ax, **kwargs)
        elif self._method_name == 'cdf':
            data.plot(kind='line', color=color, drawstyle='steps-post',
                      ax=ax, **kwargs)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))
        if vlines:
            ax.vlines(x=k, ymin=0, ymax=data.values, color=color)

        ax.set_xlabel(self._parent.x_label)

        if self._parent.y_label:
            ax.set_ylabel(self._parent.y_label)
        else:
            if self._method_name == 'pmf':
                ax.set_ylabel('P(X = k)')
            elif self._method_name == 'cdf':
                ax.set_ylabel('P(X ≤ k)')
            else:
                ax.set_ylabel(self._name)

        return ax
