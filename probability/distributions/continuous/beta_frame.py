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

    def drop(
            self,
            labels: Union[str, List[str]] = None,
            axis: int = 0,
            index: Union[str, List[str]] = None,
            columns: Union[str, List[str]] = None
    ) -> 'BetaFrame':
        """
        Drop one or more rows or columns.

        :param labels: Index or column labels to drop.
        :param axis: Whether to drop labels from the index (0 or ‘index’) or
                     columns (1 or ‘columns’).
        :param index: Alternative to specifying axis (labels, axis=0 is
                      equivalent to index=labels).
        :param columns: Alternative to specifying axis (labels, axis=1 is
                        equivalent to columns=labels).
        """
        return BetaFrame(
            data=self._data.drop(
                labels=labels,
                axis=axis,
                index=index, columns=columns
            )
        )
