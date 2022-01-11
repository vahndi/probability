from typing import Optional

from numpy.random import seed
from pandas import Series

from probability.distributions.mixins.data_mixins import \
    DataMixin, DataMinMixin, DataMaxMixin, DataMeanMixin, DataMedianMixin, \
    DataStdMixin, DataModeMixin


class Interval(
    DataMixin,
    DataMinMixin,
    DataMaxMixin,
    DataMeanMixin,
    DataMedianMixin,
    DataModeMixin,
    DataStdMixin,
    object
):

    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: pandas Series of interval data.
        """
        self._data: Series = data

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

    def __add__(self, other: float) -> 'Interval':
        """
        Return a new Interval distribution with a constant value subtracted from
        each datum.
        """
        return Interval(data=self._data + other)

    def __sub__(self, other: float) -> 'Interval':
        """
        Return a new Interval distribution with a constant value subtracted from
        each datum.
        """
        return Interval(data=self._data - other)


if __name__ == '__main__':

    s = Series([1, 2, 2, 3, 3, 3, 4, 4, 4])
    i = Interval(data=s)
    print(i.rvs(10))
