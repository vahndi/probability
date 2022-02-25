from typing import List, Union, Type, Optional

from numpy import inf, linspace
from pandas import Series, IntervalIndex, cut, qcut
from seaborn import kdeplot, histplot

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from probability.distributions import Gamma
from probability.distributions.data.ordinal import Ordinal
from probability.distributions.mixins.data_mixins import DataDistributionMixin, \
    DataMinMixin, DataMaxMixin, DataMeanMixin, DataMedianMixin, DataStdMixin, \
    DataModeMixin, DataCategoriesMixin, DataNumericMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin


class Ratio(
    DataDistributionMixin,
    DataNumericMixin,
    DataMinMixin,
    DataMaxMixin,
    DataMeanMixin,
    DataMedianMixin,
    DataStdMixin,
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
        Plot a kde plot of the distribution.

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

    def __repr__(self):

        return (
            f'{self.name}: Ratio['
            f'min={self._data.min()}, '
            f'max={self._data.max()}, '
            f'mean={self._data.mean()}'
            f']'
        )
