from typing import Union, List

from pandas import Series, DataFrame, concat


class DataMixin(object):

    _data: Series

    @property
    def data(self) -> Series:
        """
        Return the underlying data used to construct the Distribution.
        """
        return self._data


class DataMinMixin(object):

    _data: Series

    def min(self) -> Union[float, int, str]:
        """
        Return the smallest value in the data.
        """
        return self._data.min()


class DataMaxMixin(object):

    _data: Series

    def max(self) -> Union[float, int, str]:
        """
        Return the largest value in the data.
        """
        return self._data.max()


class DataMeanMixin(object):

    _data: Series

    def mean(self) -> float:
        """
        Return the mean value of the data.
        """
        return self._data.mean()


class DataMedianMixin(object):

    _data: Series

    def median(self) -> Union[int, float]:
        """
        Return the median value of the data.
        """
        return self._data.median()


class DataStdMixin(object):

    _data: Series

    def std(self) -> float:
        """
        Return the median value of the data.
        """
        return self._data.std()


class DataModeMixin(object):

    _data: Series

    def mode(self) -> Union[int, float, str,
                            List[int], List[float], List[str]]:
        """
        Return the most frequently occurring value(s) in the data.
        """
        mode = self._data.mode()
        if len(mode) > 1:
            return mode.to_list()
        else:
            return mode[0]


class CategoricalDataMixin(object):

    _data: Series

    def cpt(self, condition: 'CategoricalDataMixin') -> DataFrame:
        """
        Return the conditional probability of each category given different
        values of condition.
        """
        self_name = self._data.name
        other_name = condition._data.name
        data = concat([condition._data, self._data], axis=1)
        joint = data[[other_name, self_name]].value_counts()
        marginal = data[other_name].value_counts()
        marginal.index.name = other_name
        return (joint / marginal).unstack()
