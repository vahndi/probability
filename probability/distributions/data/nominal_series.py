from typing import Union, Dict, Any

from pandas import Series

from probability.distributions.data.nominal import Nominal
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesModeMixin
from probability.distributions.mixins.data.data_series_category_mixins import \
    DataSeriesCategoryMixin, DataSeriesCountsMixin, DataSeriesPMFsMixin, \
    DataSeriesPMFBetasMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


class NominalSeries(
    DataSeriesMixin,
    DataSeriesCategoryMixin,
    DataSeriesCountsMixin,
    DataSeriesPMFsMixin,
    DataSeriesPMFBetasMixin,
    DataSeriesModeMixin,
    object
):
    """
    Series of Nominal distributions.
    """
    pass

