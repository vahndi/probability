from typing import Union, Tuple, List

from pandas import Series

from probability.distributions.data.ordinal import Ordinal
from probability.distributions.mixins.data_mixins import DataMixin, \
    DataMinMixin, DataMaxMixin, DataMeanMixin, DataMedianMixin, DataStdMixin, \
    DataModeMixin


class Count(
    DataMixin,
    DataMinMixin,
    DataMaxMixin,
    DataMeanMixin,
    DataMedianMixin,
    DataStdMixin,
    DataModeMixin,
    object
):
    def __init__(self, data: Series):
        """
        Create a new Count distribution.

        :param data: pandas Series.
        """
        self._data: Series = data

    def as_ordinal(
            self,
            values: List[Union[int, Tuple[int, int]]]
    ):
        """
        Convert to an Ordinal variable.

        :param values: List of values or value ranges to map to categories.
                       e.g. [1, 2, (3, 5), (6, None)] will give an
                       Ordinal with categories ['1', '2', '3-5', '6+']
        """
        mapping = {}
        new_categories = []
        for value in values:
            if isinstance(value, int):
                category_name = str(value)
                new_categories.append(category_name)
                mapping[value] = category_name
            else:
                min_val = int(value[0])
                max_val = int(value[1]) if value[1] is not None else None
                if max_val is not None:
                    category_name = f'{min_val}-{max_val}'
                else:
                    category_name = f'{min_val}+'
                    max_val = int(self._data.max())
                new_categories.append(category_name)
                for val in range(min_val, max_val + 1):
                    mapping[val] = category_name
        new_data = self._data.map(mapping).astype('category')
        new_data = new_data.cat.set_categories(new_categories, ordered=True)
        return Ordinal(data=new_data)

    def __repr__(self):

        return (
            f'{self.name}: Count['
            f'min={self._data.min()}, '
            f'max={self._data.max()}, '
            f'mean={self._data.mean()}'
            f']'
        )
