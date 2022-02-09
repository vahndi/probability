from typing import List, Union

from pandas import Series

from probability.distributions.mixins.data_mixins import DataDistributionMixin, \
    DataCPTMixin, DataModeMixin, DataInformationMixin, DataCategoriesMixin


class Nominal(
    DataDistributionMixin,
    DataCategoriesMixin,
    DataModeMixin,
    DataCPTMixin,
    DataInformationMixin,
    object
):

    _ordered = False  # used for categorical mixin methods

    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: Categorical pandas Series.
        """
        self._data: Series = data
        self._categories: List[str] = data.cat.categories.to_list()

    def __repr__(self):

        cat_counts = self._data.value_counts().reindex(self._categories)
        str_cat_counts = ', '.join([
            f'"{cat}": {count}'
            for cat, count in cat_counts.items()
        ])
        return f'{self.name}: Nominal[{str_cat_counts}]'
