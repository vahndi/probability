from typing import List, Any

from pandas import Series


class DataCountsMixin(object):

    _data: Series
    _categories: List[Any]

    def counts(self) -> Series:
        """
        Return a Series with the count of each category.
        """
        value_counts = self._data.value_counts().reindex(self._categories)
        return value_counts.fillna(0)

    def pmf(self) -> Series:
        """
        Return a Series with the probability of each category.
        """
        counts = self.counts()
        return counts / counts.sum()
