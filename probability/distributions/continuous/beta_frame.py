from typing import List, Union

from pandas import DataFrame, concat

from probability.distributions.continuous.beta_series import BetaSeries
from probability.distributions.mixins.continuous_frame_mixin import \
    ContinuousFrameMixin


class BetaFrame(
    ContinuousFrameMixin,
    object
):

    def __init__(self, data: DataFrame):
        """
        Create a new BetaFrame.

        :param data: DataFrame of Beta distributions.
        """
        self._data: DataFrame = data

    @staticmethod
    def from_beta_series(data: List[BetaSeries]):

        series = concat([bs.data for bs in data], axis=1)
        return BetaFrame(series)
