from typing import TypeVar, Iterable, Optional

from numpy.random import seed
from pandas import Series, merge

DDM = TypeVar('DDM', bound='DataDistributionMixin')


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
        """
        Return the name of the Series of data.
        """
        return self._data.name

    def rename(self: DDM, name: str) -> DDM:
        """
        Rename the Series of data.
        """
        return type(self)(data=self._data.rename(name))

    def rvs(self, num_samples: int,
            random_state: Optional[int] = None) -> Series:
        """
        Sample `num_samples` random values from the distribution.
        """
        if random_state is not None:
            seed(random_state)
        return self._data.sample(
            n=num_samples, replace=True
        ).reset_index(drop=True)

    def filter_to(
            self: DDM,
            other: 'DataDistributionMixin'
    ) -> DDM:
        """
        Filter the data to the common indices with the other distribution.
        """
        left = self.data if self.data.name is not None else self.data.rename('left')
        right = other.data if other.data.name is not None else other.data.rename('right')
        merged = merge(left=left, right=right,
                       left_index=True, right_index=True)
        data = merged.iloc[:, 0].rename(self.data.name)
        return type(self)(data=data)

    def loc(
            self: DDM,
            other: Iterable
    ) -> DDM:
        """
        Similar to pandas .loc but return distribution.
        """
        return type(self)(data=self.data.loc[other])

    def __or__(self: DDM, other):
        """
        Condition this empirical distribution on `other`:
        - If `other` is a multi-category `CategoricalDistribution` (or list of
          them), returns `self.split_by(other)` (conditional DataSeries P(X|C)).
        - If `other` is a single-category `CategoricalDistribution` or another
          filtered `DataDistributionMixin`, returns `self.filter_to(other)`
          (conditioned distribution P(X|C=c)).
        - If `other` is a boolean `Series` mask, returns `self.loc(other)`.
        """
        from probability.distributions.protocols import CategoricalDistribution

        if isinstance(other, Series) and other.dtype == bool:
            return self.loc(other)
        elif isinstance(other, list):
            return self.split_by(other)
        elif isinstance(other, CategoricalDistribution):
            if other.num_categories == 1:
                return self.filter_to(other)
            else:
                return self.split_by(other)
        elif isinstance(other, DataDistributionMixin):
            return self.filter_to(other)
        else:
            raise TypeError(
                'other must be a CategoricalDistribution, '
                'List[CategoricalDistribution], DataDistributionMixin, '
                'or boolean Series'
            )

    def __len__(self):

        return len(self._data)
