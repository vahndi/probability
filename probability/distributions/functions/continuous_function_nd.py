from numpy import array, ndarray
from pandas import Series, MultiIndex
from scipy.stats import rv_continuous
from typing import overload, Iterable


class ContinuousFunctionNd(object):

    def __init__(self, distribution: rv_continuous, method_name: str, name: str,
                 num_dims: int, parent: object):
        """
        :param distribution: The scipy distribution to calculate with.
        :param method_name: The name of the method to call on the distribution.
        :param name: An intuitive name for the function.
        :param num_dims: The number of dimensions, K, of the function.
        :param parent: The parent distribution object, used to call str(...) for series labels.
        """
        self._distribution = distribution
        self._num_dims = num_dims
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: object = parent

    @overload
    def at(self, x: Iterable[float]) -> float:
        pass

    @overload
    def at(self, x: Iterable[Iterable]) -> Series:
        pass

    @overload
    def at(self, x: ndarray) -> Series:
        pass

    def at(self, x):
        """
        Evaluate the function for each value of [x1, x2, ..., xk] given as x.

        :param x: [x1, x2, ..., xk] or [[x11, x12, ..., x1k], [x21, x22, ..., x2k], ...]
        """
        x = array(x)
        if x.ndim == 1:
            return self._method(x)
        elif x.ndim == 2:
            return Series(
                index=MultiIndex.from_arrays(
                    arrays=x.T,
                    names=[f'x{num}' for num in range(1, self._num_dims + 1)]
                ), data=self._method(x), name=f'{self._name}({self._parent})'
            )
