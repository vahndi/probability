from itertools import product
from typing import List, Union, Type, Optional, Iterable, Dict

from numpy import inf, linspace, floor, ceil, arange, histogram
from pandas import Series, IntervalIndex, cut, qcut, DataFrame
from seaborn import kdeplot, histplot

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from probability.distributions import Gamma
from probability.distributions.data.ordinal import Ordinal
from probability.distributions.mixins.data.data_numeric_comparison_mixin import \
    DataNumericComparisonMixin
from probability.distributions.mixins.data.data_categories_mixin import \
    DataCategoriesMixin
from probability.distributions.mixins.data.data_distribution_mixin import \
    DataDistributionMixin
from probability.distributions.mixins.data.data_aggregate_mixins import \
    DataMinMixin, DataMaxMixin, DataMeanMixin, DataMedianMixin, DataStdMixin, \
    DataModeMixin, DataVarMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin


class Ratio(
    DataDistributionMixin,
    DataNumericComparisonMixin,
    DataMinMixin,
    DataMaxMixin,
    DataMeanMixin,
    DataMedianMixin,
    DataStdMixin,
    DataVarMixin,
    DataModeMixin,
    object
):
    def __init__(self, data: Series):
        """
        Create a new Ratio distribution.

        :param data: pandas Series.
        """
        data = data.dropna()
        self._data: Series = data

    def histogram(
            self,
            bins: Union[int, float, Iterable[float]],
            min_pct: float = 0.0,
            max_pct: float = 1.0
    ) -> Series:
        """
        Calculate a histogram of the data.

        :param bins: int number of bins, float bin spacing, or sequence of bin
                     edges.
        :param min_pct: Lowest percentile of data to use for the histogram.
        :param max_pct: Highest percentile of data to use for the histogram.
        :return: Series with indices of low, high and values of count.
        """
        if isinstance(bins, float):
            # Calculate bins for distribution.
            spacing = bins
            low_bin = spacing * floor(self._data.quantile(min_pct) / spacing)
            high_bin = spacing * ceil(self._data.quantile(max_pct) / spacing)
            # calculate bins
            bins = arange(low_bin, high_bin + spacing, spacing)
        data = self._data
        if min_pct > 0:
            min_val = self._data.quantile(min_pct)
            data = data.loc[data >= min_val]
        if max_pct < 1:
            max_val = self._data.quantile(max_pct)
            data = data.loc[data <= max_val]
        hist, bin_edges = histogram(a=data, bins=bins)
        counts = DataFrame([
            {'min': bin_edges[i],
             'max': bin_edges[i + 1],
             'count': hist[i]}
            for i in range(len(hist))
        ])
        return counts.set_index(['min', 'max'])['count']

    def as_ordinal(
            self,
            method: str,
            categories: Union[int, List[float], IntervalIndex],
            labels: Optional[List[str]] = None
    ) -> Ordinal:
        """
        Quantize the data and convert to an Ordinal.

        :param method: Pandas method to slice the data.
                       One of {'cut', 'qcut'}.
        :param categories: Number of categories (for 'cut' and 'qcut'),
                           or list of bin edges or IntervalIndex (for 'cut').
        :param labels: Optional list of labels for when method='cut'
        """
        if method == 'cut':
            data = cut(x=self.data, bins=categories, labels=labels)
        elif method == 'qcut':
            data = qcut(x=self.data, q=categories, duplicates='drop')
        else:
            raise ValueError("method must be one of {'cut', 'qcut'}")
        return Ordinal(data=data)

    def to_gamma(self) -> Gamma:

        return Gamma.fit(self._data)

    def plot_kde(
            self,
            color: Color = 'k',
            inflated_value: float = 0.0,
            inflated_threshold: float = 0.5,
            mean_line: Union[bool, dict] = False,
            median_line: Union[bool, dict] = False,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot a kde plot of the distribution.

        :param color: Color of the bars.
        :param inflated_value: Value which may occur disproportionately often in
                               the data due to a mixed distribution (e.g. ZIPF),
                               or collection method
                               (e.g. code all > 100% as 101%).
        :param inflated_threshold: Proportion of values which must equal the
                                   inflated_value to add an additional line to
                                   plot the non-inflated curve.
        :param mean_line: Boolean flag to indicate whether to annotate the mean.
                          Or dict of kws to pass to AxesFormatter.add_v_line.
        :param median_line: Boolean flag to indicate whether to annotate the
                            mean.
                            Or dict of kws to pass to AxesFormatter.add_v_line.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        # main kde
        kdeplot(
            data=self._data,
            color=color,
            ax=axf.axes,
            label=self.name
        )
        # non-inflated kde
        non_zero = self._data.loc[self._data != inflated_value]
        if len(non_zero) / len(self._data) > inflated_threshold:
            kdeplot(
                data=non_zero,
                color=color,
                ls='--',
                ax=axf.axes,
                label=f'{self.name} != {inflated_value}'
            )
        # draw mean
        mean_line_kws = None
        if isinstance(mean_line, bool):
            if mean_line is True:
                mean_line_kws = {
                    'color': 'green',
                    'line_style': ':'
                }
        else:
            mean_line_kws = mean_line
        if mean_line_kws is not None:
            mean = self._data.mean()
            axf.add_v_line(x=mean, **mean_line_kws)
            y_min, y_max = axf.get_y_min(), axf.get_y_max()
            axf.add_text(x=mean, y=y_min + 0.8 * (y_max - y_min),
                         text=f'mean = {mean: 0.1f}')
        # draw median
        median_line_kws = None
        if isinstance(median_line, bool):
            if median_line is True:
                median_line_kws = {
                    'color': 'blue',
                    'line_style': ':'
                }
        else:
            median_line_kws = median_line
        if median_line_kws is not None:
            median = self._data.median()
            axf.add_v_line(x=median, **median_line_kws)
            y_min, y_max = axf.get_y_min(), axf.get_y_max()
            axf.add_text(x=median, y=y_min + 0.7 * (y_max - y_min),
                         text=f'median = {median: 0.1f}')
        return axf

    def plot_hist(
            self,
            color: Color = 'k',
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot a histogram of the distribution.

        :param color: Color of the bars.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        histplot(
            data=self._data,
            color=color,
            ax=axf.axes
        )
        return axf

    def plot_conditional_dist_densities(
            self,
            categorical: Union[DataDistributionMixin, DataCategoriesMixin],
            fit_dist: Type[RVContinuous1dMixin],
            hdi: float = 0.95,
            width: float = 0.8,
            num_segments: int = 100,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            color_median: Optional[Color] = None,
            edge_color: Optional[Color] = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot conditional probability densities of the data, split by the
        categories of an Ordinal or Nominal distribution.

        :param categorical: Nominal or Ordinal distribution.
        :param fit_dist: Continuous distribution to fit to each set of
                         conditioned data.
        :param hdi: Highest Density Interval width for each distribution.
        :param width: Width of each density bar.
        :param num_segments: Number of segments to plot per density.
        :param color: Color for the densest part of each distribution.
        :param color_min: Color for the sparsest part of each distribution,
                          if different to color.
        :param color_mean: Color for mean data markers.
        :param color_median: Color for median data markers.
        :param edge_color: Optional color for the edge of each density bar.
        :param axf: Optional AxesFormatter to plot on.
        """
        axf = axf or AxesFormatter()
        cats = categorical.categories
        n_cats = len(cats)
        yy_min, yy_max = inf, -inf
        # filter categorical data
        shared_ix = list(
            set(self._data.index).intersection(categorical.data.index)
        )
        cat_data = categorical.data.loc[shared_ix]
        ratio_data = self._data.loc[shared_ix]
        for c, category in enumerate(categorical.categories):
            cat_ratio_data = ratio_data.loc[cat_data == category]
            if len(cat_ratio_data) == 0:
                continue
            # fit distribution and find limits for HDI
            cat_dist = fit_dist.fit(data=cat_ratio_data)
            y_min, y_max = cat_dist.hdi(hdi)
            yy_min, yy_max = min(y_min, yy_min), max(y_max, yy_max)
            # plot density
            axf.add_v_density(
                x=c + 1,
                y_to_z=cat_dist.pdf().at(
                    linspace(y_min, y_max, num_segments + 1)),
                color=color, color_min=color_min, edge_color=edge_color,
                width=width
            )
            # plot descriptive statistics lines
            if color_mean is not None:
                mean = cat_ratio_data.mean()
                axf.add_line(x=[c + 0.55, c + 1.45], y=[mean, mean],
                             color=color_mean)
            if color_median is not None:
                median = cat_ratio_data.median()
                axf.add_line(x=[c + 0.55, c + 1.45], y=[median, median],
                             color='g')
        # labels
        axf.set_text(
            title=f'{hdi: .0%} HDIs of p({self.name}|{categorical.name})',
            x_label=categorical.name,
            y_label=self.name
        )
        # axes
        axf.set_x_lim(0, n_cats + 1)
        yy_range = yy_max - yy_min
        axf.set_y_lim(yy_min - yy_range * 0.05, yy_max + yy_range * 0.05)
        axf.x_ticks.set_locations(range(1, n_cats + 1)).set_labels(cats)

        return axf

    def cohens_d(self, other: 'Ratio') -> float:
        """
        Calculate the Cohen's d standardized difference of means between self
        and other.

        https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
        """
        n1, n2 = len(self), len(other)
        v1, v2 = self._data.var(), other._data.var()
        x1, x2 = self._data.mean(), other._data.mean()
        # pooled standard deviation
        s = (
            ((n1 - 1) * v1 + (n2 - 1) * v2) /
            (n1 + n2 - 2)
        ) ** 0.5
        return (x1 - x2) / s

    def conditional_cohens_d(
            self,
            categorical: DataCategoriesMixin
    ) -> DataFrame:
        """
        Return a matrix of the Cohen's d of the Ratio conditioned on each
        category compared to it conditioned on each other category.

        :param categorical: Categorical data distribution to condition on.

        https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
        """
        dists: Dict[str, Ratio] = {
            category: self.filter_to(
                categorical.keep(category)
            ).rename(category)
            for category in categorical.categories
        }
        v: Dict[str, float] = {
            category: dist.var()
            for category, dist in dists.items()
        }
        n: Dict[str, int] = {
            category: len(dist)
            for category, dist in dists.items()
        }
        x: Dict[str, float] = {
            category: dist.mean()
            for category, dist in dists.items()
        }
        results = []
        for control, treatment in product(dists.keys(), dists.keys()):
            n1, n2 = n[control], n[treatment]
            v1, v2 = v[control], v[treatment]
            x1, x2 = x[control], x[treatment]
            s = (
                    ((n1 - 1) * v1 + (n2 - 1) * v2) /
                    (n1 + n2 - 2)
            ) ** 0.5
            results.append({
                'control': control,
                'treatment': treatment,
                'd': (x2 - x1) / s
            })
        return DataFrame(results).pivot(
            index='control', columns='treatment', values='d'
        ).reindex(dists.keys(), axis=0).reindex(dists.keys(), axis=1)

    def __repr__(self):

        return (
            f'{self.name}: Ratio['
            f'min={self._data.min()}, '
            f'max={self._data.max()}, '
            f'mean={self._data.mean()}'
            f']'
        )
