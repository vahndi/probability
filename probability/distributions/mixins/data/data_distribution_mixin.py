from typing import TypeVar, Iterable

from pandas import Series, merge

DDM = TypeVar('DDM', bound='DataDistributionMixin')


class DataDistributionMixin(object):

    _data: Series

    @property
    def data(self) -> Series:
        """
        Return the underlying data used to construct the Distribution.
        """
        return self._data

    @property
    def name(self):
        """
        Return the name of the Series of data.
        """
        return self._data.name

    def rename(self: DDM, name: str) -> DDM:
        """
        Rename the Series of data.
        """
        return type(self)(data=self._data.rename(name))

    def filter_to(
            self: DDM,
            other: 'DataDistributionMixin'
    ) -> DDM:
        """
        Filter the data to the common indices with the other distribution.
        """
        merged = merge(left=self.data, right=other.data,
                       left_index=True, right_index=True)
        data = merged.iloc[:, 0]
        return type(self)(data=data)

    def loc(
            self: DDM,
            other: Iterable
    ) -> DDM:
        """
        Similar to pandas .loc but return distribution.
        """
        return type(self)(data=self.data.loc[other])

    def __len__(self):

        return len(self._data)