from typing import overload, Iterable

from pandas import Series, DataFrame


class ContinuousFunction1dSeries(object):

    def __init__(self, data: Series):

        self._data: Series = data

    @overload
    def at(self, x: float) -> Series:
        pass

    @overload
    def at(self, x: Iterable) -> DataFrame:
        pass

    def at(self, x):
        """
        Log of the probability density function of the given RV.
        """
        if isinstance(x, float) or isinstance(x, int):
            return self._data.map(lambda d: d.at(x))
        elif isinstance(x, Iterable):
            return DataFrame(
                data=[d.at(x) for d in self._data.values],
                index=self._data.index,
                columns=x
            )

    def __len__(self) -> int:

        return len(self._data)

    def __str__(self):

        return str(self._data)
