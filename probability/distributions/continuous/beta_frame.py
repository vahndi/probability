from typing import List, Optional, Union

from matplotlib.patches import Patch

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from numpy import linspace, arange
from pandas import DataFrame

from probability.distributions import Beta
from probability.models.utils import loop_variable


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
            stagger: bool = True,
            item_width: float = 0.8,
            min_pct: float = 0.025, max_pct: float = 0.975,
            z_max: Optional[Union[float, List[float]]] = None,
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
        :param min_pct: Min ppf to start plotting at for each distribution.
        :param max_pct: Max ppf to end plotting at for each distribution.
        :param z_max: Optional normalizing constant to divide each bar's height
                      by.
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
                y_to_z = beta.pdf().at(linspace(
                    beta.ppf().at(min_pct), beta.ppf().at(max_pct), 96
                ))
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
