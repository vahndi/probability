from pandas import Series
from typing import Iterable, overload


class ContinuousFunction(object):

    def __init__(self, distribution, method_name: str, name: str):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)

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

    def plot(self, at: Iterable):

        pass
