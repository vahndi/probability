from typing import List

from pandas import Series

from probability.distributions.mixins.data_mixins import DataMixin


class Boolean(DataMixin):

    def __init__(self, data: Series):
        """
        Create a new Boolean distribution.
        """
        self._data: Series = data

    @property
    def categories(self) -> List[bool]:

        return [False, True]

    def __repr__(self):

        value_counts = self._data.value_counts()
        str_value_counts = ', '.join([
            f'"{value}": {count}'
            for value, count in value_counts.items()
        ])
        return f'{self.name}: Boolean[{str_value_counts}]'
