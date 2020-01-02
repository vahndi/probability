from pandas import Series
from typing import Iterable, overload


class DiscreteFunction(object):

    def __init__(self, distribution, method_name: str, name: str):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)

    @overload
    def at(self, k: int) -> int:
        pass

    @overload
    def at(self, k: Iterable) -> Series:
        pass

    def at(self, k):
        """
        Log of the probability density function of the given RV.
        """
        if isinstance(k, int):
            return self._method(k)
        elif isinstance(k, Iterable):
            return Series(index=k, data=self._method(k), name=self._name)

    def plot(self, at: Iterable):

        pass
