from typing import List, Union

from pandas import Series

from probability.distributions.mixins.data_mixins import DataMixin, \
    DataCPTMixin, DataModeMixin, DataMutualInformationMixin


class Nominal(
    DataMixin,
    DataModeMixin,
    DataCPTMixin,
    DataMutualInformationMixin,
    object
):

    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: Categorical pandas Series.
        """
        self._data: Series = data
        self._categories: List[str] = data.cat.categories.to_list()

    def drop(self, categories: Union[str, List[str]]) -> 'Nominal':
        """
        Drop one or more categories from the underlying data.

        :param categories: Categories to drop.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[~self._data.isin(categories)]
        new_cats = [cat for cat in self._categories if cat not in categories]
        data = data.cat.set_categories(
            new_categories=new_cats,
            ordered=False
        )
        return Nominal(data=data)

    def keep(self, categories: Union[str, List[str]]) -> 'Nominal':
        """
        Drop all the categories from the data not in the one(s) given.

        :param categories: Categories to keep.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[self._data.isin(categories)]
        data = data.cat.set_categories(
            new_categories=categories,
            ordered=False
        )
        return Nominal(data=data)
