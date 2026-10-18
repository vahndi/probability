from itertools import product
from typing import Union, Optional, List

from matplotlib.axes import Axes
from numpy import inf, arange, nan
from numpy.random import choice
from pandas import Series, DataFrame, concat
from seaborn import heatmap

from probability.distributions import Ordinal
from probability.plotting import (
    Color,
    add_rectangle,
    add_text,
    cross_fade,
    format_as_percent,
    new_axes,
)
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesModeMixin
from probability.distributions.mixins.data.data_series_category_mixins import \
    DataSeriesCategoryMixin, DataSeriesCountsMixin, DataSeriesPMFsMixin, \
    DataSeriesPMFBetasMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin, DataSeriesTrendMixin


class OrdinalSeries(
    DataSeriesMixin,
    DataSeriesTrendMixin,
    DataSeriesCategoryMixin,
    DataSeriesCountsMixin,
    DataSeriesPMFsMixin,
    DataSeriesPMFBetasMixin,
    DataSeriesModeMixin,
    object
):
    """
    Series of Ordinal distributions.
    """

    def correlation(self) -> DataFrame:
        """
        Return a DataFrame with the rank correlation of each Ordinal with itself
        and each other.
        """
        correlations = [
            {'x': x, 'y': y, 'ρ': self[x].correlation(self[y])}
            for x, y in product(self.keys(), self.keys())
        ]
        return DataFrame(
            correlations
        ).set_index(['x', 'y']).unstack('y').droplevel(0, axis=1)

    def entropy(self) -> Series:
        """
        Return a Series with the entropy of each item.
        """
        return Series({
            k: v.entropy()
            for k, v in self._data.items()
        })

    def unsplit(self) -> Ordinal:
        """
        Return an Ordinal with all the data for each Ordinal in the
        OrdinalSeries concatenated.
        """
        return Ordinal(
            data=concat([
                self._data[key].data for key in self.keys()
            ]).astype('category').cat.set_categories(
                self._data.iloc[0].data.cat.categories,
                ordered=True
            )
        )

    def _sample_series(
            self,
            unsplit_dist: Ordinal,
            n_x: Series
    ) -> 'OrdinalSeries':
        p_y = unsplit_dist.pmf()
        y = p_y.index.to_list()
        s_x = {
            k: Ordinal(Series(
                data=choice(a=y, size=n_x[k], p=p_y),
            ).astype('category').cat.set_categories(
                self.categories, ordered=True
            ))
            for k in self.keys()
        }
        return OrdinalSeries(s_x)

    def plot_density_grid(
            self,
            width: float = 0.8,
            heights: float = 0.9,
            color: Union[Color, List[Color]] = 'k',
            pct_labels: bool = True,
            edges: bool = False,
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            color_median: Optional[Color] = None,
            ax: Optional[Axes] = None
    ) -> Axes:
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
        :param ax: Optional matplotlib Axes to plot on.
        """
        ax = ax or new_axes()
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
        # assign colors
        if isinstance(color, list):
            colors = {key: color for key, color in zip(self.keys(), color)}
        else:
            colors = {key: color for key in self.keys()}
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
                add_rectangle(
                    ax=ax,
                    width=bar_width, height=heights,
                    x_left=x_center - bar_width / 2,
                    y_bottom=y_center - heights / 2,
                    color=rect_color,
                    alpha=pct / max_pct,
                    line_width=0
                )
                if pct_labels:
                    add_text(
                        ax=ax,
                        x=x_center, y=y_center,
                        text=format_as_percent(pct, 1),
                        h_align='center', v_align='center',
                        bbox_edge_color='k', bbox_fill=True,
                        bbox_face_color='white'
                    )
            if len(key_ord_data) == 0:
                continue
            # plot descriptive statistics lines
            if color_mean is not None:
                interval = 1 + key_ord_data.cat.codes
                mean = interval.mean()
                ax.plot([k + 0.55, k + 1.45], [mean, mean], color=color_mean)
            if color_median is not None:
                interval = 1 + key_ord_data.cat.codes
                median = interval.median()
                ax.plot([k + 0.55, k + 1.45], [median, median], color='g')
            if edges:
                x_center = 1 + k
                add_rectangle(
                    ax=ax,
                    width=bar_width,
                    height=y_max - y_min,
                    x_left=x_center - bar_width / 2,
                    y_bottom=y_min,
                    edge_color=(
                        colors[key] if color_min is None else
                        cross_fade(color_min, colors[key], 0.5)
                    ),
                    fill=False
                )
        # labels
        ax.set_title(f'Distributions of p({self.name}|{self.index.name})')
        ax.set_xlabel(self.index.name)
        ax.set_ylabel(str(self.name))
        # axes
        ax.set_xlim(0.5, n_keys + 0.5)
        ax.set_ylim(0, n_cats + 1)
        ax.set_xticks(range(1, n_keys + 1))
        ax.set_xticklabels(keys)
        ax.set_yticks(range(1, n_cats + 1))
        ax.set_yticklabels(cats)

        return ax

    def plot_correlation(
            self,
            grid_font_size: float = 12,
            fmt: str = '.1%',
            ax: Optional[Axes] = None,
            **heatmap_kwargs
    ) -> Axes:
        """
        Plot a correlation grid for the Ordinals in the Series.

        :param grid_font_size: Font size for the grid labels.
        :param fmt: Formatter for the grid labels.
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        corr = self.correlation().round(2)
        corr.values[[arange(corr.shape[0])] * 2] = nan
        heatmap(
            corr, ax=ax,
            annot=True, fmt=fmt, annot_kws={'fontsize': grid_font_size},
            **heatmap_kwargs
        )
        return ax

    def plot_pmf_lines(
            self,
            ax: Optional[Axes] = None,
            **plot_kwargs
    ) -> Axes:

        pmf_data = concat([
            self[key].pmf().rename(key)
            for key in self.keys()
        ], axis=1)
        ax = ax or new_axes()
        pmf_data.plot.line(ax=ax, **plot_kwargs)
        return ax
