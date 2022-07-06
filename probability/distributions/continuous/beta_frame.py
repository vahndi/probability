from typing import List, Union

from pandas import DataFrame, concat, Series

from probability.distributions.continuous.beta_series import BetaSeries
from probability.distributions.mixins.continuous_frame_mixin import \
    ContinuousFrameMixin


class BetaFrame(
    ContinuousFrameMixin,
    object
):

    def __init__(
            self,
            data: Union[DataFrame, Series]
    ):
        """
        Create a new BetaFrame.

        :param data: DataFrame of Beta distributions, or Series of BetaSeries.
        """
        if isinstance(data, DataFrame):
            self._data: DataFrame = data
        elif isinstance(data, Series):
            self._data = DataFrame({
                key: data[key].data
                for key in data.keys()
            }).T
        else:
            raise TypeError('must pass DataFrame or Series of BetaSeries')

    @property
    def T(self) -> 'BetaFrame':

        return BetaFrame(self._data.T)

    @staticmethod
    def from_beta_series(data: List[BetaSeries]):

        series = concat([bs.data for bs in data], axis=1)
        return BetaFrame(series)
