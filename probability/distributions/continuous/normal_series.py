from typing import Union, Mapping

from pandas import Series, DataFrame

from probability.distributions import Normal
from probability.distributions.mixins.continuous_series_mixin import \
    ContinuousSeriesMixin


class NormalSeries(
    ContinuousSeriesMixin,
    object
):

    def __init__(self, data: Series):
        """
        Create a new NormalSeries.

        :param data: Series of Normal distributions.
        """
        self._data: Series = data

    @staticmethod
    def from_ratio_frame(
            data: DataFrame,
            name: str = ''
    ):
        """
        Create a new NormalSeries using the distributions of data in each
        column of a DataFrame.

        :param data: Data with ratio values.
        :param name: Name for the Series.
        """
        normals = {}
        for col in data.columns:
            normals[col] = Normal(
                mu=data[col].mean(),
                sigma=data[col].std()
            )
        normals = Series(data=normals, name=name)
        return NormalSeries(normals)

    @property
    def data(self) -> Union[Series, Mapping[str, Normal]]:
        return self._data
