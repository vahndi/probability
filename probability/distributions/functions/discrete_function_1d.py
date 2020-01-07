from matplotlib.axes import Axes
from pandas import Series
from scipy.stats import rv_discrete
from typing import Iterable, overload

from probability.plots import new_axes


class DiscreteFunction1d(object):

    def __init__(self, distribution: rv_discrete, method_name: str, name: str,
                 parent: object):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: object = parent

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

    def plot(self, k: Iterable[int], color: str = 'C0', ax: Axes = None) -> Axes:

        data: Series = self.at(k)
        ax = ax or new_axes()
        if self._name == 'PMF':
            data.plot(kind='line', label=str(self._parent), color=color,
                      marker='o', linestyle='-', ax=ax)
        elif self._name == 'CDF':
            data.plot(kind='line', label=str(self._parent), color=color,
                      marker='o', linestyle='-', drawstyle='steps-post', ax=ax)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))
        ax.set_xlabel('k')
        ax.set_ylabel(self._name)
        return ax