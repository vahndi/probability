from typing import List

from pandas import DataFrame, concat

from probability.distributions.continuous.beta_series import BetaSeries
from probability.distributions.mixins.continuous_frame_mixin import \
    ContinuousFrameMixin


class NormalFrame(
    ContinuousFrameMixin,
    object
):

    def __init__(self, data: DataFrame):
        """
        Create a new NormalFrame.

        :param data: DataFrame of Normal distributions.
        """
        self._data: DataFrame = data

    @staticmethod
    def from_normal_series(data: List[BetaSeries]):

        series = concat([bs.data for bs in data], axis=1)
        return NormalFrame(series)
