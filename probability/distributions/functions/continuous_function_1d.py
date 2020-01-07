from matplotlib.axes import Axes
from pandas import Series
from scipy.stats import rv_continuous
from typing import Iterable, overload

from probability.plots import new_axes


class ContinuousFunction1d(object):

    def __init__(self, distribution: rv_continuous, method_name: str, name: str,
                 parent: object):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: object = parent

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

    def plot(self, x: Iterable, color: str = 'C0', ax: Axes = None) -> Axes:

        data: Series = self.at(x)
        ax = ax or new_axes()
        if self._name in ('PDF', 'CDF'):
            data.plot(kind='line', label=str(self._parent), color=color, ax=ax)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))
        ax.set_xlabel('x')
        ax.set_ylabel(self._name)
        return ax
