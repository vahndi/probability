from typing import Union, Dict, Any

from pandas import Series

from probability.distributions import Nominal
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
    def __init__(self, data: Union[Series, Dict[Any, Nominal]]):
        """
        Create a new NominalSeries.

        :param data: Series or dict of Nominal distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data
