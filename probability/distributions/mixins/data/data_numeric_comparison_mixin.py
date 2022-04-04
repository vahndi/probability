from typing import TypeVar, Union

from pandas import Series


DNCM = TypeVar('DNCM', bound='DataNumericComparisonMixin')


class DataNumericComparisonMixin(object):

    _data: Series

    def where_eq(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data == value])

    def where_ne(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data != value])

    def where_gt(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data > value])

    def where_lt(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data < value])

    def where_ge(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data >= value])

    def where_le(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data <= value])
