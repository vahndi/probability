from typing import Optional, Union, List

from numpy.random import seed
from pandas import Series

from probability.distributions.mixins.rv_mixins import RVS1dMixin


class Interval(
    RVS1dMixin,
    object
):

    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: pandas Series of interval data.
        """
        self._data: Series = data

    @property
    def data(self) -> Series:

        return self._data

    def rvs(self, num_samples: int,
            random_state: Optional[int] = None) -> Series:
        """
        Sample `num_samples` random values from the distribution.
        """
        if random_state is not None:
            seed(random_state)
        return self._data.sample(
            n=num_samples, replace=True
        ).reset_index(drop=True)

    def median(self) -> Union[int, float]:

        return self._data.median()

    def mode(self) -> Union[int, float, List[int], List[float]]:

        mode = self._data.mode()
        if len(mode) > 1:
            return mode.to_list()
        else:
            return mode[0]

    def mean(self) -> float:

        return self._data.mean()

    def std(self) -> float:

        return self._data.std()

    def __add__(self, other: float) -> 'Interval':
        """
        Return a new Interval distribution with a constant value subtracted from
        each datum.
        """
        return Interval(
            data=self._data + other
        )

    def __sub__(self, other: float) -> 'Interval':
        """
        Return a new Interval distribution with a constant value subtracted from
        each datum.
        """
        return Interval(
            data=self._data - other
        )


if __name__ == '__main__':

    s = Series([1, 2, 2, 3, 3, 3, 4, 4, 4])
    i = Interval(data=s)
    print(i.rvs(10))
