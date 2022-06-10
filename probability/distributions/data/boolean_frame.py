from typing import Optional

from pandas import DataFrame

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
from mpl_format.utils.number_utils import format_as_percent
from probability.distributions import Boolean


class BooleanFrame(object):

    def __init__(self, data: DataFrame):
        """
        Create a new BooleanFrame.

        :param data: DataFrame of Boolean distributions.
        """
        self._data: DataFrame = data

    @property
    def data(self) -> DataFrame:
        return self._data

    def means(self) -> DataFrame:

        return self._data.applymap(lambda x: x.mean())

    def lens(self) -> DataFrame:

        return self._data.applymap(lambda x: len(x))

    def plot_bars(
            self,
            width: float = 0.8,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            pct_labels: bool = True,
            edges: bool = False,
            conditional: bool = True,
            axf: Optional[AxesFormatter] = None
    ):
        """
        Plot percentage of each entry that is True.

        :param width: Width of each density bar.
        :param color: Color for the densest part of each distribution.
        :param color_min: Color for the sparsest part of each distribution,
                          if different to color.
        :param pct_labels: Whether to add percentage labels to each density bar.
        :param edges: Whether to plot the edges of each set of bars.
        :param conditional: Whether to use the max of all means or the max of
                            means of each index to color the bars.
        :param axf: Optional AxesFormatter to plot on.
        """
        axf = axf or AxesFormatter()
        if not conditional:
            max_mean = self.means().max().max()
        width_ratios = self.lens() / self.lens().max(axis=0).mean()
        for x, ix in enumerate(self._data.index):
            bar_width = width_ratios.loc[ix] * width
            if conditional:
                max_mean = self._data.loc[ix].apply(lambda d: d.mean()).max()
            for y, col in enumerate(self._data.columns):
                dist: Boolean = self._data.loc[ix, col]
                pct = dist.mean()
                if color_min is not None:
                    rect_color = cross_fade(color_min, color, pct / max_mean)
                else:
                    rect_color = color
                x_center = x + 1
                y_center = y + 1
                axf.add_rectangle(
                    width=bar_width,
                    height=1,
                    x_center=x_center,
                    y_center=y_center,
                    color=rect_color,
                    alpha=pct / max_mean,
                    line_width=0
                )
                if pct_labels:
                    axf.add_text(x=x_center, y=y_center,
                                 text=format_as_percent(pct, 1),
                                 h_align='center', v_align='center',
                                 bbox_edge_color='k', bbox_fill=True,
                                 bbox_face_color='white')
            if edges:
                x_center = x + 1
                axf.add_rectangle(
                    width=bar_width,
                    height=len(self.data.columns),
                    x_left=x_center - bar_width / 2,
                    y_bottom=0.5,
                    edge_color=(
                        color if color_min is None else
                        cross_fade(color_min, color, 0.5)
                    ),
                    fill=False
                )
        axf.set_x_lim(0.5, len(self._data.index) + 0.5)
        axf.set_y_lim(0, len(self._data.columns) + 1)
        axf.x_ticks.set_locations(
            range(1, len(self._data.index) + 1)
        ).set_labels(self._data.index.to_list())
        axf.y_ticks.set_locations(
            range(1, len(self._data.columns) + 1)
        ).set_labels(self._data.columns.to_list())
        axf.set_text(
            title=f'Distribution of '
                  f'p({self.data.columns.name}|{self.data.index.name})',
            x_label=self.data.index.name,
            y_label=self.data.columns.name
        )
        return axf
