from typing import Optional, Mapping, Union, List, Any, Dict

from numpy import linspace
from numpy.ma import arange
from pandas import Series, DataFrame

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
from probability.distributions import Beta
from probability.distributions.functions.continuous_function_1d_series import \
    ContinuousFunction1dSeries
from probability.distributions.mixins.continuous_series_mixin import \
    ContinuousSeriesMixin
from probability.models.utils import loop_variable


class BetaSeries(
    ContinuousSeriesMixin,
    object
):

    def __init__(self, data: Union[Series, Dict[Any, Beta]]):
        """
        Create a new BetaSeries.

        :param data: Series of Beta distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data

    @staticmethod
    def from_bool_frame(
            data: DataFrame,
            prior_alpha: float = 0,
            prior_beta: float = 0,
            name: str = ''
    ):
        """
        Create a new BetaSeries using the counts of True and False or 1 and 0
        in a DataFrame.

        :param data: Data with True / False counts.
        :param prior_alpha: Value for alpha assuming these represent posterior
                            distributions.
        :param prior_beta: Value for alpha assuming these represent posterior
                            distributions.
        :param name: Name for the Series.
        """
        betas = {}
        for col in data.columns:
            betas[col] = Beta(
                alpha=prior_alpha + (data[col] == 1).sum(),
                beta=prior_beta + (data[col] == 0).sum()
            )
        betas = Series(data=betas, name=name)
        return BetaSeries(betas)

    @staticmethod
    def from_proportions(data: DataFrame):
        """
        Fit to a DataFrame of proportions. Returns a Series with one item for
        each column in data.
        """
        return BetaSeries(Series({
            column: Beta.fit(data[column].dropna())
            for column in data.columns
        }))

    @property
    def data(self) -> Union[Series, Mapping[str, Beta]]:
        return self._data
