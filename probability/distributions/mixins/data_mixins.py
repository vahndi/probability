from typing import Union, List

from numpy import log
from pandas import Series, DataFrame, concat
from scipy.stats import entropy


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


class DataCPTMixin(object):

    _data: Series

    def cpt(self, condition: 'DataCPTMixin') -> DataFrame:
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


class DataMutualInformationMixin(object):

    _data: Series

    def mutual_information(self, other: 'DataMutualInformationMixin') -> float:
        """
        Return the mutual information between self and other.

        https://en.wikipedia.org/wiki/Mutual_information#In_terms_of_PMFs_for_discrete_distributions
        """
        x_counts = self._data.value_counts().rename_axis('x')
        y_counts = other._data.value_counts().rename_axis('y')
        xy_counts = concat([
            self._data.rename('x'),
            other._data.rename('y')
        ], axis=1).value_counts()
        p_x = x_counts / x_counts.sum()
        p_y = y_counts / y_counts.sum()
        p_xy = xy_counts / xy_counts.sum()
        calc = p_xy.rename('p(x,y)').to_frame()
        calc['p(x)'] = p_x.loc[
            calc.index.get_level_values('x')].to_list()
        calc['p(y)'] = p_y.loc[
            calc.index.get_level_values('y')].to_list()
        calc['I(x,y)'] = calc['p(x,y)'] * (
                calc['p(x,y)'] / (calc['p(x)'] * calc['p(y)'])
        ).map(log)
        return calc['I(x,y)'].sum()

    def entropy(self) -> float:
        """
        Return the entropy of the distribution (self-information).
        """
        return entropy(self._data.value_counts())

    def relative_mutual_information(self, other: 'DataMutualInformationMixin'):
        """
        Return the proportion of entropy in self explained by observing
        other.
        """
        return self.mutual_information(other) / self.entropy()
