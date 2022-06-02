from typing import Union, Dict, Any

from pandas import Series

from probability.distributions import Boolean
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesModeMixin, DataSeriesMedianMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


class BooleanSeries(
    DataSeriesMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesModeMixin,
    DataSeriesMedianMixin,
    object
):
    """
    Series or dict of Boolean distributions.
    """
    def __init__(self, data: Union[Series, Dict[Any, Boolean]]):
        """
        Create a new BooleanSeries.

        :param data: Series of Boolean distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data
