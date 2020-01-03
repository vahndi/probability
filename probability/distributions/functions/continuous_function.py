from matplotlib.axes import Axes
from pandas import Series
from typing import Iterable, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from probability.distributions.mixins.rv_continuous_mixin import RVContinuousMixin
from probability.plots import new_axes


class ContinuousFunction(object):

    def __init__(self, distribution, method_name: str, name: str,
                 parent: 'RVContinuousMixin'):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: 'RVContinuousMixin' = parent

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
        data.plot(kind='line', label=str(self._parent), color=color)
        ax.set_xlabel('x')
        ax.set_ylabel(self._name)
        return ax
