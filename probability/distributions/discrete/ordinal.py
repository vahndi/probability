from typing import List

from numpy.random import choice
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
        self._cat_to_num = {
            category: ix
            for ix, category in enumerate(self._categories)
        }

    def rvs(self, num_samples: int) -> Series:
        """
        Sample `num_samples` random values from the distribution.
        """
        return self._data.sample(
            n=num_samples, replace=True
        ).reset_index(drop=True)

    def __gt__(self, other: 'Ordinal'):

        if not isinstance(other, Ordinal):
            raise TypeError('Can only compare an Ordinal with another Ordinal')
        self_samples = self.rvs(NUM_SAMPLES_COMPARISON)
        other_samples = other.rvs(NUM_SAMPLES_COMPARISON)
        self_values = self_samples.replace(self._cat_to_num)
        other_values = other_samples.replace(other._cat_to_num)
        return (self_values > other_values).mean()

    def __lt__(self, other: 'Ordinal'):

        if not isinstance(other, Ordinal):
            raise TypeError('Can only compare an Ordinal with another Ordinal')
        self_samples = self.rvs(NUM_SAMPLES_COMPARISON)
        other_samples = other.rvs(NUM_SAMPLES_COMPARISON)
        self_values = self_samples.replace(self._cat_to_num)
        other_values = other_samples.replace(other._cat_to_num)
        return (self_values < other_values).mean()

    def __eq__(self, other: 'Ordinal'):

        if not isinstance(other, Ordinal):
            raise TypeError('Can only compare an Ordinal with another Ordinal')
        self_samples = self.rvs(NUM_SAMPLES_COMPARISON)
        other_samples = other.rvs(NUM_SAMPLES_COMPARISON)
        self_values = self_samples.replace(self._cat_to_num)
        other_values = other_samples.replace(other._cat_to_num)
        return (self_values == other_values).mean()
