from typing import List, Optional, Tuple

from numpy.random import seed
from pandas import Series

from probability.distributions.mixins.rv_mixins import RVS1dMixin, \
    NUM_SAMPLES_COMPARISON


class Ordinal(
    RVS1dMixin,
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
        if random_state is not None:
            seed(random_state)
        return self._data_vals.sample(
            n=num_samples, replace=True
        ).reset_index(drop=True)

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

    def median(self) -> str:

        median_ix = self._data_vals.median()
        return self._val_to_name[median_ix]

    def mode(self) -> Series:

        return self._data.mode()

    def __repr__(self):

        cat_counts = self._data.value_counts().reindex(self._categories)
        str_cat_counts = ', '.join([
            f'"{cat}": {count}'
            for cat, count in cat_counts.items()
        ])
        return f'Ordinal[{str_cat_counts}]'
