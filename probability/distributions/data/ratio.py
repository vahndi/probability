from typing import List, Union

from pandas import Series, IntervalIndex, cut, qcut

from probability.distributions import Gamma
from probability.distributions.data.ordinal import Ordinal
from probability.distributions.mixins.data_mixins import DataMixin, \
    DataMinMixin, DataMaxMixin, DataMeanMixin, DataMedianMixin, DataStdMixin, \
    DataModeMixin


class Ratio(
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
        Create a new Ratio distribution.

        :param data: pandas Series.
        """
        self._data: Series = data

    def as_ordinal(
            self,
            method: str,
            categories: Union[int, List[float], IntervalIndex]
    ) -> Ordinal:
        """
        Quantize the data and convert to an Ordinal.

        :param method: Pandas method to slice the data.
                       One of {'cut', 'qcut'}.
        :param categories: Number of categories (for 'cut' and 'qcut'),
                           or list of bin edges or IntervalIndex (for 'cut').
        """
        if method == 'cut':
            data = cut(self.data, categories)
        elif method == 'qcut':
            data = qcut(x=self.data, q=categories, duplicates='drop')
        else:
            raise ValueError("method must be one of {'cut', 'qcut'}")
        return Ordinal(data=data)

    def to_gamma(self) -> Gamma:

        return Gamma.fit(self._data)

    def __repr__(self):

        return (
            f'{self.name}: Ratio['
            f'min={self._data.min()}, '
            f'max={self._data.max()}, '
            f'mean={self._data.mean()}'
            f']'
        )
