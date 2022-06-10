from typing import Union, Dict, Any, Optional, Mapping, List

from numpy import inf
from pandas import Series

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
from mpl_format.utils.number_utils import format_as_percent
from probability.distributions import Ordinal
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesModeMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


class OrdinalSeries(
    DataSeriesMixin,
    DataSeriesModeMixin,
    object
):
    """
    Series of Ordinal distributions.
    """
    def __init__(self, data: Union[Series, Dict[Any, Ordinal]]):
        """
        Create a new OrdinalSeries. Assumes all distributions have the same
        categories.

        :param data: Series or dict of Ordinal distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Union[Series, Mapping[Any, Ordinal]] = data

    @property
    def categories(self) -> List[str]:

        return self._data.iloc[0].categories

    def plot_densities(
            self,
            width: float = 0.8,
            heights: float = 0.9,
            color: Color = 'k',
            pct_labels: bool = True,
            edges: bool = False,
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            color_median: Optional[Color] = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot conditional probability densities of the data.

        :param width: Width of each density bar.
        :param heights: Height of each density bar.
        :param color: Color for the densest part of each distribution.
        :param pct_labels: Whether to add percentage labels to each density bar.
        :param edges: Whether to plot the edges of each set of bars.
        :param color_min: Color for the sparsest part of each distribution,
                          if different to color.
        :param color_mean: Color for mean data markers.
        :param color_median: Color for median data markers.
        :param axf: Optional AxesFormatter to plot on.
        """
        axf = axf or AxesFormatter()
        keys = self.keys()
        n_keys = len(keys)
        cats = self.categories
        n_cats = len(cats)
        # find highest number of total items in a distribution
        max_cat_sum = max([
            self._data[key].counts().sum()
            for key in self.keys()
        ])
        # find highest % of total items of a category in its distribution
        max_pct = max([
            self._data[key].counts().max() / self._data[key].counts().sum()
            for key in self.keys()
        ])
        for k, key in enumerate(self.keys()):
            key_ord_data = self[key]
            value_counts = key_ord_data.data.value_counts().reindex(
                key_ord_data.categories).fillna(0)
            cat_sum = value_counts.sum()
            y_min, y_max = inf, -inf
            for i, (item, count) in enumerate(value_counts.items()):
                pct = count / cat_sum
                if color_min is not None:
                    rect_color = cross_fade(color_min, color, pct / max_pct)
                else:
                    rect_color = color
                bar_width = width * cat_sum / max_cat_sum
                x_center = 1 + k
                y_center = 1 + i
                if i == 0:
                    y_min = y_center - heights / 2
                if i == len(value_counts) - 1:
                    y_max = y_center + heights / 2
                axf.add_rectangle(
                    width=bar_width, height=heights,
                    x_left=x_center - bar_width / 2,
                    y_bottom=y_center - heights / 2,
                    color=rect_color,
                    alpha=pct / max_pct,
                    line_width=0
                )
                if pct_labels:
                    axf.add_text(x=x_center, y=y_center,
                                 text=format_as_percent(pct, 1),
                                 h_align='center', v_align='center',
                                 bbox_edge_color='k', bbox_fill=True,
                                 bbox_face_color='white')
            if len(key_ord_data) == 0:
                continue
            # plot descriptive statistics lines
            if color_mean is not None:
                interval = 1 + key_ord_data.cat.codes
                mean = interval.mean()
                axf.add_line(x=[k + 0.55, k + 1.45], y=[mean, mean],
                             color=color_mean)
            if color_median is not None:
                interval = 1 + key_ord_data.cat.codes
                median = interval.median()
                axf.add_line(x=[k + 0.55, k + 1.45], y=[median, median],
                             color='g')
            if edges:
                x_center = 1 + k
                axf.add_rectangle(
                    width=bar_width,
                    height=y_max - y_min,
                    x_left=x_center - bar_width / 2,
                    y_bottom=y_min,
                    edge_color=(
                        color if color_min is None else
                        cross_fade(color_min, color, 0.5)
                    ),
                    fill=False
                )
        # labels
        axf.set_text(
            title=f'Distributions of p({self.name}|{self.index.name})',
            x_label=self.index.name,
            y_label=str(self.name)
        )
        # axes
        axf.set_x_lim(0.5, n_keys + 0.5)
        axf.set_y_lim(0, n_cats + 1)
        axf.x_ticks.set_locations(range(1, n_keys + 1)).set_labels(keys)
        axf.y_ticks.set_locations(
            range(1, n_cats + 1)).set_labels(cats)

        return axf
