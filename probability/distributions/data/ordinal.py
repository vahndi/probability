from typing import List, Optional, Tuple, Union

from numpy.random import seed
from pandas import Series

from probability.distributions.data.interval import Interval
from probability.distributions.mixins.data_mixins import \
    DataMixin, DataCPTMixin, DataMinMixin, DataMaxMixin, \
    DataInformationMixin
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class Ordinal(
    DataMixin,
    DataMinMixin,
    DataMaxMixin,
    DataCPTMixin,
    DataInformationMixin,
    object
):
    """
    Ordinal data is a categorical, statistical data type where the variables
    have natural, ordered categories and the distances between the categories
    are not known. These data exist on an ordinal scale, one of four levels of
    measurement described by S. S. Stevens in 1946. The ordinal scale is
    distinguished from the nominal scale by having a ranking. It also differs
    from the interval scale and ratio scale by not having category widths that
    represent equal increments of the underlying attribute.

    https://en.wikipedia.org/wiki/Ordinal_data
    """
    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: Categorical pandas Series.
        """
        self._data: Series = data
        self._categories: List[str] = data.cat.categories.to_list()
        self._name_to_val = {
            category: ix
            for ix, category in enumerate(self._categories)
        }
        self._val_to_name = {
            ix: category
            for ix, category in enumerate(self._categories)
        }
        self._data_vals: Series = self._data.replace(self._name_to_val)

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

    def rvs_values(self, num_samples: int,
                   random_state: Optional[int] = None) -> Series:
        """
        Sample `num_samples` random values from the distribution.
        """
        return self._data_vals.sample(
            n=num_samples, replace=True,
            random_state=random_state
        ).reset_index(drop=True)

    def median_value(self) -> int:

        return self._data_vals.median()

    def median(self) -> str:

        return self._val_to_name[self.median_value()]

    def mode(self) -> Union[str, List[str]]:

        mode = self._data_vals.mode()
        if len(mode) > 1:
            return mode.to_list()
        else:
            return mode[0]

    def as_interval(self) -> Interval:

        return Interval(data=self._data_vals)

    def drop(self, categories: Union[str, List[str]]) -> 'Ordinal':
        """
        Drop one or more categories from the underlying data.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[~self._data.isin(categories)]
        new_cats = [cat for cat in self._categories if cat not in categories]
        data = data.cat.set_categories(
            new_categories=new_cats, ordered=True
        )
        return Ordinal(data=data)

    def keep(self, categories: Union[str, List[str]]) -> 'Ordinal':
        """
        Drop all the categories from the data not in the one(s) given.

        :param categories: Categories to keep.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[self._data.isin(categories)]
        new_cats = [cat for cat in self._categories if cat in categories]
        data = data.cat.set_categories(
            new_categories=new_cats,
            ordered=True
        )
        return Ordinal(data=data)

    def _check_can_compare(self, other: 'Ordinal'):

        if not isinstance(other, Ordinal):
            raise TypeError('Can only compare an Ordinal with another Ordinal')
        if not self._categories == other._categories:
            raise ValueError('Both Ordinals must have the same categories.')

    def _comparison_samples(
            self, other: 'Ordinal'
    ) -> Tuple[Series, Series]:

        self_samples = self.rvs_values(NUM_SAMPLES_COMPARISON)
        other_samples = other.rvs_values(NUM_SAMPLES_COMPARISON)
        return self_samples, other_samples

    def __eq__(self, other: 'Ordinal') -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values == other_values).mean()

    def __ne__(self, other: 'Ordinal') -> float:

        return 1 - (self == other)

    def __lt__(self, other: 'Ordinal') -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values < other_values).mean()

    def __gt__(self, other: 'Ordinal') -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values > other_values).mean()

    def __le__(self, other: 'Ordinal') -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values <= other_values).mean()

    def __ge__(self, other: 'Ordinal') -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values >= other_values).mean()

    def __repr__(self):

        cat_counts = self._data.value_counts().reindex(self._categories)
        str_cat_counts = ', '.join([
            f'"{cat}": {count}'
            for cat, count in cat_counts.items()
        ])
        return f'Ordinal[{str_cat_counts}]'
