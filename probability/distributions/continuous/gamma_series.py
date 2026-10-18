from typing import Union, Dict, Any, Mapping

from pandas import Series

from probability.distributions.continuous.gamma import Gamma
from probability.distributions.mixins.continuous_series_mixin import (
    ContinuousSeriesMixin
)


class GammaSeries(
    ContinuousSeriesMixin,
    object
):
    """
    Series or dict of Gamma distributions.
    """

    def __init__(self, data: Union[Series, Dict[Any, Gamma]]):
        """
        Create a new GammaSeries.

        :param data: Series of Gamma distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data

    def to_dict(self) -> Dict[str, Gamma]:
        return {k: v for k, v in self._data.items()}

    @property
    def data(self) -> Union[Series, Mapping[str, Gamma]]:
        return self._data
