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


class DataSeriesMeanMixin(object):

    _data: Series

    def mean(self) -> Series:
        """
        Return a Series of the mean value in each distribution.
        """
        return Series({
            ix: self._data[ix].mean()
            for ix in self._data.index
        })


class DataSeriesModeMixin(object):

    _data: Series

    def mode(self) -> Series:
        """
        Return a Series of the mode value in each distribution.
        """
        return Series({
            ix: self._data[ix].mode()
            for ix in self._data.index
        })


class DataSeriesMedianMixin(object):

    _data: Series

    def median(self) -> Series:
        """
        Return a Series of the median value in each distribution.
        """
        return Series({
            ix: self._data[ix].median()
            for ix in self._data.index
        })


class DataSeriesStdMixin(object):

    _data: Series

    def std(self) -> Series:
        """
        Return a Series of the standard deviation of each distribution.
        """
        return Series({
            ix: self._data[ix].std()
            for ix in self._data.index
        })
