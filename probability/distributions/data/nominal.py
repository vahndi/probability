from typing import List, Union, Optional

from numpy import nan
from pandas import Series

from probability.distributions.data.boolean import Boolean
from probability.distributions.data.ordinal import Ordinal
from probability.distributions.mixins.data_mixins import \
    DataDistributionMixin, \
    DataProbabilityTableMixin, DataModeMixin, DataInformationMixin, DataCategoriesMixin, \
    DataDiscreteMixin


class Nominal(
    DataDistributionMixin,
    DataCategoriesMixin,
    DataDiscreteMixin,
    DataModeMixin,
    DataProbabilityTableMixin,
    DataInformationMixin,
    object
):

    _ordered = False  # used for categorical mixin methods

    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: Categorical pandas Series.
        """
        data = data.dropna()
        self._data: Series = data
        self._categories: List[str] = data.cat.categories.to_list()

    def as_ordinal(
            self,
            ordered_categories: List[str],
    ) -> Ordinal:
        """
        Convert to an Ordinal distribution with the given categories.
        Any categories not given will be dropped.

        :param ordered_categories: List of ordered categories.
        """
        data = self.drop([
            c for c in self.categories
            if c not in ordered_categories
        ]).data
        data = data.cat.set_categories(ordered_categories, ordered=True)
        return Ordinal(data=data)

    def as_boolean(
            self,
            false: Union[str, List[str]],
            true: Union[str, List[str]],
            empty: Optional[Union[str, List[str]]] = None
    ) -> Boolean:
        """
        Convert to a Boolean distribution.

        :param false: Categories to map to False.
        :param true: Categories to map to True.
        :param empty: Categories to map to nan.
        """
        if not isinstance(false, list):
            false = [false]
        if not isinstance(true, list):
            true = [true]
        if empty is None:
            empty = []
        elif not isinstance(empty, list):
            empty = [empty]
        if not set(true + false + empty) == set(self.categories):
            raise ValueError('Must provide all categories in the distribution')
        data = self._data.copy()
        for f in false:
            data = data.replace(f, False)
        for t in true:
            data = data.replace(t, True)
        for e in empty:
            data = data.replace(e, nan)
        data = data.dropna()
        return Boolean(data)

    def __repr__(self):

        cat_counts = self._data.value_counts().reindex(self._categories)
        str_cat_counts = ', '.join([
            f'"{cat}": {count}'
            for cat, count in cat_counts.items()
        ])
        return f'{self.name}: Nominal[{str_cat_counts}]'
