from typing import List, Optional, Union

from matplotlib.patches import Patch

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from numpy import linspace, arange, log
from pandas import DataFrame, concat

from probability.distributions import Beta
from probability.distributions.continuous.beta_series import BetaSeries
from probability.models.utils import loop_variable


class BetaFrame(object):

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

    @property
    def data(self) -> DataFrame:
        return self._data

    def plot_density_bars(
            self,
            color: Union[Color, List[Color]] = 'k',
            color_min: Optional[Union[Color, List[Color]]] = None,
            group_width: float = 0.8,
            stagger: bool = True,
            item_width: float = 0.8,
            hdi: float = 0.95,
            resolution: int = 100,
            z_max: Optional[Union[float, List[float]]] = None,
            log_z: bool = False,
            min_n: int = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot each row of distributions as a group of density bars.

        :param color: Color for each column, or all bars.
        :param color_min: Min color for each column, or all bars.
        :param group_width: Width of each column group.
        :param item_width: Width of each item as a proportion of the group
                           width divided by the number of items per group.
        :param stagger: Whether to plot items within a row next to each
                        other.
        :param hdi: Highest Density Interval width for each distribution.
        :param z_max: Optional normalizing constant to divide each bar's height
                      by.
        :param resolution: Number of density elements per unit y.
        :param log_z: Whether to take the log of z before plotting.
        :param min_n: Minimum number of pseudo-observations (α + β) to plot a
                      distribution.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        n_rows = self._data.shape[0]
        n_cols = self._data.shape[1]
        if not stagger:
            width_per_item = group_width
            item_centers = [0] * n_cols
        else:
            width_per_item = group_width * item_width / n_cols
            item_centers = linspace(
                -group_width / 2 + width_per_item / 2,
                group_width / 2 - width_per_item / 2,
                n_cols
            )
        color = loop_variable(color, n_cols)
        color_min = loop_variable(color_min, n_cols)
        # add distributions
        beta: Beta
        for i_row, (row_name, betas) in enumerate(self._data.iterrows()):
            for i_col, (col_name, beta) in enumerate(betas.items()):
                if min_n is not None and beta.alpha + beta.beta < min_n:
                    continue
                y_min, y_max = beta.hdi(hdi)
                n_bars = round(resolution * (y_max - y_min))
                y_to_z = beta.pdf().at(linspace(
                    y_min, y_max,
                    n_bars + 1
                ))
                if log_z:
                    y_to_z = y_to_z.map(log)
                axf.add_v_density(
                    x=i_row + 1 + item_centers[i_col],
                    y_to_z=y_to_z,
                    color=color[i_col], color_min=color_min[i_col],
                    width=width_per_item,
                    z_max=z_max,
                    h_align='center'
                )
        # axes
        axf.set_x_lim(0, n_rows + 1)
        axf.x_ticks.set_locations(range(1, n_rows + 1))
        axf.x_ticks.set_labels(self._data.index)
        axf.y_ticks.set_locations(arange(0, 1.1, 0.1))
        axf.set_y_lim(-0.05, 1.05)
        # legend
        patches = []
        for i_col in range(n_cols):
            patches.append(Patch(
                color=color[i_col],
                label=self._data.columns[i_col]
            ))
        axf.axes.legend(handles=patches)
        return axf

    def __str__(self):

        return str(self._data)
