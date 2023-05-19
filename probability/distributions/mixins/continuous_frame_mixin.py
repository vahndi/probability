from typing import Union, List, Optional, TypeVar

from matplotlib.patches import Patch
from numpy import arange, linspace, log
from pandas import DataFrame

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from probability.distributions.mixins.rv_mixins import PDF1dMixin, \
    PPFContinuous1dMixin
from probability.models.utils import loop_variable


CFM = TypeVar('CFM', bound='ContinuousFrameMixin')


class ContinuousFrameMixin(object):

    _data: DataFrame

    @property
    def data(self) -> DataFrame:
        return self._data

    def means(self) -> DataFrame:
        return self._data.applymap(lambda dist: dist.mean())

    def drop(
            self: CFM,
            labels: Union[str, List[str]] = None,
            axis: int = 0,
            index: Union[str, List[str]] = None,
            columns: Union[str, List[str]] = None
    ) -> CFM:
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
        return type(self)(
            data=self._data.drop(
                labels=labels,
                axis=axis,
                index=index, columns=columns
            )
        )

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
            axf: Optional[AxesFormatter] = None,
            legend_kwargs: Optional[dict] = None
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
        :param axf: Optional AxesFormatter instance.
        :param legend_kwargs: Optional kwargs to pass to legend constructor.
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
        dist: Union[PDF1dMixin, PPFContinuous1dMixin]
        for i_row, (row_name, betas) in enumerate(self._data.iterrows()):
            for i_col, (col_name, dist) in enumerate(betas.items()):
                y_min, y_max = dist.hdi(hdi)
                n_bars = round(resolution * (y_max - y_min))
                y_to_z = dist.pdf().at(linspace(
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
        # legend
        patches = []
        for i_col in range(n_cols):
            patches.append(Patch(
                color=color[i_col],
                label=self._data.columns[i_col]
            ))
        if legend_kwargs is None:
            legend_kwargs = {}
        axf.axes.legend(handles=patches, **legend_kwargs)
        return axf

    def __str__(self):
        return str(self._data)

    def __gt__(self: CFM, other: CFM) -> DataFrame:
        """
        Return the probability that each distribution in this frame is greater
        than the corresponding distribution in the other frame.
        """
        if set(self.data.index) != set(other.data.index):
            raise IndexError('mismatched Indexes')
        if set(self.data.columns) != set(other.data.columns):
            raise IndexError('mismatched Columns')
        gt = self.data.copy()
        for ix in self.data.index:
            for col in self.data.columns:
                gt.loc[ix, col] = (
                        self.data.loc[ix, col] > other.data.loc[ix, col]
                )
        return gt

    def __lt__(self: CFM, other: CFM) -> DataFrame:
        """
        Return the probability that each distribution in this frame is greater
        than the corresponding distribution in the other frame.
        """
        return 1 - self.__gt__(other)
