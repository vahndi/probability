from types import Union
from typing import List, Optional

from pandas import DataFrame

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color


class BetaFrame(object):

    def __init__(self, data: DataFrame):
        """
        Create a new BetaSeries.

        :param data: DataFrame of Beta distributions.
        """
        self._data: DataFrame = data

    def plot_density_bars(
            self,
            color: Union[Color, List[Color]],
            color_min: Optional[Union[Color, List[Color]]] = None,
            group_width: float = 0.8,
            item_width: float = 0.8,
            width: Union[float, List[float]] = 0.8,
            min_pct: float = 0.025, max_pct: float = 0.975,
            z_max: Optional[Union[float, List[float]]] = None,
            edges: bool = False,
            axf: Optional[AxesFormatter] = None
    ):
        """
        Plot each row of distributions as a group of density bars.

        :param color: Color for each column, or all bars.
        :param color_min: Min color for each column, or all bars.
        :param group_width: Width of each column group.
        :param item_width: Width of each item as a proportion of the group
                           width.
        :param min_pct: Min ppf to start plotting at for each distribution.
        :param max_pct: Max ppf to end plotting at for each distribution.
        :param z_max: Optional normalizing constant to divide each bar's height
                      by.
        :param edges: Whether to plot the edges of each bar.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        n_rows = self._data.shape[0]
        n_cols = self._data.shape[1]

        for i_row, (row_name, betas) in enumerate(self._data.iterrows()):
            for i_col, (col_name, beta) in betas.items():
                pass
