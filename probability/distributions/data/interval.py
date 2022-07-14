from typing import Optional, Union, TYPE_CHECKING

from numpy import arange
from numpy.random import seed, choice, uniform
from pandas import Series

from probability.distributions.mixins.data.data_categories_mixin import \
    DataCategoriesMixin
from probability.distributions.mixins.data.data_comparison_mixins import \
    DataCohensDMixin
from probability.distributions.mixins.data.data_numeric_comparison_mixin import \
    DataNumericComparisonMixin
from probability.distributions.mixins.data.data_distribution_mixin import \
    DataDistributionMixin
from probability.distributions.mixins.data.data_aggregate_mixins import \
    DataMinMixin, DataMaxMixin, \
    DataMeanMixin, DataMedianMixin, DataModeMixin, \
    DataStdMixin, DataVarMixin

if TYPE_CHECKING:
    from probability.distributions.data.interval_series import IntervalSeries


class Interval(
    DataDistributionMixin,
    DataNumericComparisonMixin,
    DataMinMixin,
    DataMaxMixin,
    DataMeanMixin,
    DataMedianMixin,
    DataModeMixin,
    DataStdMixin,
    DataVarMixin,
    DataCohensDMixin,
    object
):

    def __init__(self, data: Series):
        """
        Create a new Interval distribution.

        :param data: pandas Series of interval data.
        """
        data = data.dropna()
        self._data: Series = data

    @staticmethod
    def random(
            name: str,
            min_value: Union[int, float],
            max_value: Union[int, float],
            interval: Union[int, float],
            size: int
    ):
        """
        Generate an interval distribution where the probability of observing each
        value comes from a uniform distribution.

        :param name: Name for the distribution and underlying data Series.
        :param min_value: Lowest value to generate.
        :param max_value: Highest value to generate.
        :param interval: Size of the interval between each value.
        :param size: Total number of samples to generate.
        """
        a = arange(start=min_value, stop=max_value + interval, step=interval)
        p = Series(uniform(size=len(a)))
        p /= p.sum()
        data = Series(
            data=choice(a=a, p=p, size=size),
            name=name
        )
        return Interval(data=data)

    @staticmethod
    def random_uniform(
            name: str,
            min_value: Union[int, float],
            max_value: Union[int,float],
            interval: Union[int, float],
            size: int
    ) -> 'Interval':
        """
        Generate an interval distribution with uniformly distributed
        values.

        :param name: Name for the distribution and underlying data Series.
        :param min_value: Lowest value to generate.
        :param max_value: Highest value to generate.
        :param interval: Size of the interval between each value.
        :param size: Total number of samples to generate.
        """
        a = arange(start=min_value, stop=max_value + interval, step=interval)
        data = Series(
            data=choice(a=a, size=size),
            name=name
        )
        return Interval(data=data)

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

    def split_by(
            self,
            categorical: Union[DataCategoriesMixin, DataDistributionMixin]
    ) -> 'IntervalSeries':
        """
        Split into an IntervalSeries on different values of the given categorical
        distribution.

        :param categorical: Distribution to split on
        """
        intervals_dict = {}
        for category in categorical.categories:
            intervals_dict[category] = self.filter_to(
                categorical.keep(category))
        from probability.distributions.data.interval_series \
            import IntervalSeries
        interval_series_data = Series(intervals_dict, name=self.name)
        interval_series_data.index.name = categorical.name
        return IntervalSeries(interval_series_data)

    def __add__(self, other: float) -> 'Interval':
        """
        Return a new Interval distribution with a constant value subtracted from
        each datum.
        """
        return Interval(data=self._data + other)

    def __sub__(self, other: float) -> 'Interval':
        """
        Return a new Interval distribution with a constant value subtracted from
        each datum.
        """
        return Interval(data=self._data - other)

    def __repr__(self):

        return (
            f'{self.name}: Interval['
            f'min={self._data.min()}, '
            f'max={self._data.max()}, '
            f'mean={self._data.mean()}'
            f']'
        )
