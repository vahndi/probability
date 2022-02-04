from typing import Union, List

from numpy import log
from pandas import Series, DataFrame, concat
from scipy.stats import entropy


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
        return self._data.name

    def rename(self, name: str) -> 'DataDistributionMixin':

        return type(self)(data=self._data.rename(name))


class DataCategoriesMixin(object):

    _categories: List[Union[bool, str]]

    @property
    def categories(self) -> list:

        return self._categories


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


class DataInformationMixin(object):

    _data: Series

    def _calc_frame(self, other: 'DataInformationMixin') -> DataFrame:
        """
        Calculate the joint and marginal probability distributions of self and
        other and return as a DataFrame.
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
        calc['p(x)'] = p_x.reindex(
            calc.index.get_level_values('x')).to_list()
        calc['p(y)'] = p_y.reindex(
            calc.index.get_level_values('y')).to_list()
        return calc

    def entropy(self) -> float:
        """
        Return the entropy of the distribution (self-information).
        """
        return entropy(self._data.value_counts())

    def mutual_information(self, other: 'DataInformationMixin') -> float:
        """
        In probability theory and information theory, the mutual information
        (MI) of two random variables is a measure of the mutual dependence
        between the two variables. More specifically, it quantifies the
        "amount of information" (in units such as shannons (bits), nats or
        hartleys) obtained about one random variable by observing the other
        random variable. The concept of mutual information is intimately linked
        to that of entropy of a random variable, a fundamental notion in
        information theory that quantifies the expected "amount of information"
        held in a random variable.

        https://en.wikipedia.org/wiki/Mutual_information
        """
        calc = self._calc_frame(other)
        calc['I(x,y)'] = calc['p(x,y)'] * (
                calc['p(x,y)'] / (calc['p(x)'] * calc['p(y)'])
        ).map(log)
        return calc['I(x,y)'].sum()

    def conditional_entropy(self, other: 'DataInformationMixin') -> float:
        """
        In information theory, the conditional entropy quantifies the amount of
        information needed to describe the outcome of a random variable Y given
        that the value of another random variable X is known. Here, information
        is measured in shannons, nats, or hartleys. The entropy of Y conditioned
        on X is written as H(Y|X)}.

        https://en.wikipedia.org/wiki/Conditional_entropy
        """
        calc = self._calc_frame(other)
        calc['H(x|y)'] = calc['p(x,y)'] * (
                calc['p(x,y)'] / calc['p(x)']
        ).map(log)
        return -calc['H(x|y)'].sum()

    def joint_entropy(self, other: 'DataInformationMixin') -> float:
        """
        In information theory, joint entropy is a measure of the uncertainty
        associated with a set of variables.

        https://en.wikipedia.org/wiki/Joint_entropy
        """
        calc = self._calc_frame(other)
        calc['H(x,y)'] = calc['p(x,y)'] * calc['p(x,y)'].map(log)
        return -calc['H(x,y)'].sum()

    def relative_mutual_information(self, other: 'DataInformationMixin'):
        """
        Return the proportion of entropy in self explained by observing
        other.
        """
        return self.mutual_information(other) / self.entropy()
