from itertools import product
from typing import Union, Dict, Any, Optional, Mapping, List

from numpy import inf, arange, nan
from numpy.random import choice
from pandas import Series, DataFrame, concat
from seaborn import heatmap
from tqdm import tqdm

from mpl_format.axes import AxesFormatter
from mpl_format.axes.axis_utils import new_axes
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
from mpl_format.utils.number_utils import format_as_percent
from probability.distributions import Ordinal
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesModeMixin
from probability.distributions.mixins.data.data_series_category_mixins import \
    DataSeriesCategoryMixin, DataSeriesCountsMixin, DataSeriesPMFsMixin, \
    DataSeriesPMFBetasMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


class OrdinalSeries(
    DataSeriesMixin,
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
    def __init__(self, data: Union[Series, Dict[Any, Ordinal]]):
        """
        Create a new OrdinalSeries. Assumes all distributions have the same
        categories.

        :param data: Series or dict of Ordinal distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Union[Series, Mapping[Any, Ordinal]] = data

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

    def increase_probs(self) -> Series:
        """
        Return a Series of probabilities of whether each adjacent pair of
        Ordinals increases from one to the next.
        """
        keys = self.keys()
        results = []
        for k in range(len(keys) - 1):
            k_x = keys[k]
            k_y = keys[k + 1]
            results.append({
                'x': k_x,
                'y': k_y,
                'p(y > x)': self._data[k_y].probably_greater_than(
                    self._data[k_x]
                )
            })
        return DataFrame(results).set_index(['x', 'y'])['p(y > x)']

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

    def p_increasing(
            self,
            n_iter: int = 1_000
    ) -> float:
        """
        Return the probability that the Ordinals increase as the key increases.
        """
        ref_probs = self.increase_probs()
        ref_sum = ref_probs.sum()
        ref_prod = ref_probs.product()
        # calculate p(y)
        p_y = self.unsplit().pmf()
        y = p_y.index.to_list()
        # calculate n[X]
        n_x = self.lens()
        # repeat
        results = []
        for _ in tqdm(range(n_iter)):
            # create sampled distribution for each x
            s_x = {}
            for k in self.keys():
                s_x[k] = Ordinal(Series(
                    data=choice(a=y, size=n_x[k], p=p_y),
                ).astype('category').cat.set_categories(
                    self.categories, ordered=True
                ))
            ord_test = OrdinalSeries(s_x)
            # find probability that sampled distribution is increasing,
            # and by how much
            test_probs = ord_test.increase_probs()
            test_sum = test_probs.sum()
            test_prod = test_probs.prod()
            # if result is more extreme than observed, record a 1
            if ref_sum > test_sum and ref_prod > test_prod:
                results.append(1)
            else:
                results.append(0)
        return Series(results).mean()

    def p_decreasing(
            self,
            n_iter: int = 1_000
    ) -> float:
        """
        Return the probability that the Ordinals decrease as the key increases.
        """
        ref_probs = self.increase_probs()
        ref_sum = ref_probs.sum()
        ref_prod = ref_probs.product()
        # calculate p(y)
        p_y = self.unsplit().pmf()
        y = p_y.index.to_list()
        # calculate n[X]
        n_x = self.lens()
        # repeat
        results = []
        for _ in tqdm(range(n_iter)):
            # create sampled distribution for each x
            s_x = {}
            for k in self.keys():
                s_x[k] = Ordinal(Series(
                    data=choice(a=y, size=n_x[k], p=p_y),
                ).astype('category').cat.set_categories(
                    self.categories, ordered=True
                ))
            ord_test = OrdinalSeries(s_x)
            # find probability that sampled distribution is increasing,
            # and by how much
            test_probs = ord_test.increase_probs()
            test_sum = test_probs.sum()
            test_prod = test_probs.prod()
            # if result is more extreme than observed, record a 1
            if ref_sum < test_sum and ref_prod < test_prod:
                results.append(1)
            else:
                results.append(0)
        return Series(results).mean()

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
                        colors[key] if color_min is None else
                        cross_fade(color_min, colors[key], 0.5)
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

    def plot_correlation(
            self,
            grid_font_size: float = 12,
            fmt: str = '.1%',
            axf: Optional[AxesFormatter] = None,
            **heatmap_kwargs
    ) -> AxesFormatter:
        """
        Plot a correlation grid for the Ordinals in the Series.

        :param grid_font_size: Font size for the grid labels.
        :param fmt: Formatter for the grid labels.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or new_axes()
        corr = self.correlation().round(2)
        corr.values[[arange(corr.shape[0])] * 2] = nan
        heatmap(
            corr, ax=axf.axes,
            annot=True, fmt=fmt, annot_kws={'fontsize': grid_font_size},
            **heatmap_kwargs
        )
        return axf

    def plot_pmf_lines(
            self,
            axf: Optional[AxesFormatter] = None,
            **plot_kwargs
    ):

        pmf_data = concat([
            self[key].pmf().rename(key)
            for key in self.keys()
        ], axis=1)
        axf = axf or AxesFormatter()
        pmf_data.plot.line(ax=axf.axes, **plot_kwargs)
        axf.set_text(
            title=f'p({self.name}|{self.index.name})',
            x_label=str(self.name), y_label=f'p({self.name})'
        )
        return axf

    def plot_pmf_bars(
            self,
            axf: Optional[AxesFormatter] = None,
            **plot_kwargs
    ):

        pmf_data = concat([
            self[key].pmf().rename(key)
            for key in self.keys()
        ], axis=1)
        axf = axf or AxesFormatter()
        pmf_data.plot.bar(ax=axf.axes, **plot_kwargs)
        axf.set_text(
            title=f'p({self.name}|{self.index.name})',
            x_label=str(self.name), y_label=f'p({self.name})'
        )
        return axf
