from typing import List, Hashable

from pandas import Series, Index


class DataSeriesMixin(object):

    _data: Series

    @property
    def index(self) -> Index:
        return self._data.index

    @property
    def name(self) -> Hashable:
        return self._data.name

    def keys(self) -> List[str]:
        return self._data.keys().to_list()

    def __getitem__(self, item):
        """
        Return the distribution corresponding to the given key.
        """
        return self._data[item]
