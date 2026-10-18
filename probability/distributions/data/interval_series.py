from typing import Union, Dict, Any

from pandas import Series

from probability.distributions import Interval
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesStdMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


class IntervalSeries(
    DataSeriesMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesStdMixin,
    object
):
    """
    Series or dict of Interval distributions.
    """
    pass

