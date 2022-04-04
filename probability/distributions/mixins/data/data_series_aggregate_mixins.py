from pandas import Series


class DataSeriesMinMixin(object):

    _data: Series

    def min(self) -> Series:
        """
        Return a Series of the smallest value in each distribution.
        """
        return Series({
            ix: self._data[ix].min()
            for ix in self._data.index
        })


class DataSeriesMaxMixin(object):

    _data: Series

    def max(self) -> Series:
        """
        Return a Series of the largest value in each distribution.
        """
        return Series({
            ix: self._data[ix].max()
            for ix in self._data.index
        })


class DataSeriesPMFMixin(object):

    _data: Series

