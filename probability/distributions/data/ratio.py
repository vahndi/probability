from itertools import product
from typing import List, Union, Type, Optional, Iterable, Dict, TYPE_CHECKING, \
    Tuple, Any

from matplotlib.axes import Axes
from numpy import inf, linspace, floor, ceil, arange, histogram, ndarray
from numpy.random import seed
from pandas import Series, IntervalIndex, cut, qcut, DataFrame
from seaborn import kdeplot, histplot

from probability.distributions import Gamma, Normal
from probability.plotting import Color, add_v_density, new_axes
from probability.distributions.mixins.data.data_comparison_mixins import \
    DataCohensDMixin, DataHypothesisTestMixin, DataProbComparisonMixin

from probability.distributions.mixins.data.data_numeric_comparison_mixin import \
    DataNumericComparisonMixin
from probability.distributions.mixins.data.data_categories_mixin import \
    DataCategoriesMixin
from probability.distributions.mixins.data.data_distribution_mixin import \
    DataDistributionMixin
from probability.distributions.mixins.data.data_aggregate_mixins import \
    DataMinMixin, DataMaxMixin, DataMeanMixin, DataMedianMixin, DataStdMixin, \
    DataModeMixin, DataVarMixin, DataSemMixin
from probability.distributions.mixins.data.data_sortable_mixin import \
    DataSortableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.protocols import CategoricalDistribution

if TYPE_CHECKING:
    from probability.distributions.data.ordinal import Ordinal


class Ratio(
    DataDistributionMixin,
    DataProbComparisonMixin,
    DataNumericComparisonMixin,
    DataMinMixin,
    DataMaxMixin,
    DataMeanMixin,
    DataMedianMixin,
    DataStdMixin,
    DataVarMixin,
    DataSemMixin,
    DataModeMixin,
    DataSortableMixin,
    DataCohensDMixin,
    DataHypothesisTestMixin,
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
    ) -> 'Ordinal':
        """
        Quantize the data and convert to an Ordinal.

        :param method: Pandas method to slice the data.
                       One of {'cut', 'qcut'}.
        :param categories: Number of categories (for 'cut' and 'qcut'),
                           or list of bin edges or IntervalIndex (for 'cut').
        :param labels: Optional list of labels for when method='cut'
        """
        from probability.distributions.data.ordinal import Ordinal
        if method == 'cut':
            data = cut(x=self.data, bins=categories, labels=labels)
        elif method == 'qcut':
            data = qcut(x=self.data, q=categories, duplicates='drop')
        else:
            raise ValueError("method must be one of {'cut', 'qcut'}")
        return Ordinal(data=data)

    def to_gamma(self) -> Gamma:
        """
        Fit a gamma distribution to the data.
        """
        return Gamma.fit(data=self._data)

    def to_normal(self) -> Normal:
        """
        Fit a normal distribution to the data.
        """
        return Normal.fit(data=self._data)

    def plot_kde(
            self,
            color: Color = 'k',
            inflated_value: float = 0.0,
            inflated_threshold: float = 0.5,
            mean_line: Union[bool, dict] = False,
            median_line: Union[bool, dict] = False,
            ax: Optional[Axes] = None
    ) -> Axes:
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
        :param mean_line: Boolean flag to indicate whether to annotate the mean,
                          or dict of kws to pass to Axes.axvline.
        :param median_line: Boolean flag to indicate whether to annotate the
                            median, or dict of kws to pass to Axes.axvline.
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        # main kde
        kdeplot(
            data=self._data,
            color=color,
            ax=ax,
            label=self.name
        )
        # non-inflated kde
        non_zero = self._data.loc[self._data != inflated_value]
        if len(non_zero) / len(self._data) > inflated_threshold:
            kdeplot(
                data=non_zero,
                color=color,
                ls='--',
                ax=ax,
                label=f'{self.name} != {inflated_value}'
            )
        # draw mean
        mean_line_kws = None
        if isinstance(mean_line, bool):
            if mean_line is True:
                mean_line_kws = {
                    'color': 'green',
                    'linestyle': ':'
                }
        else:
            mean_line_kws = dict(mean_line)
            if 'line_style' in mean_line_kws:
                mean_line_kws['linestyle'] = mean_line_kws.pop('line_style')
            if 'line_width' in mean_line_kws:
                mean_line_kws['linewidth'] = mean_line_kws.pop('line_width')
        if mean_line_kws is not None:
            mean = self._data.mean()
            ax.axvline(x=mean, **mean_line_kws)
            y_min, y_max = ax.get_ylim()
            ax.text(x=mean, y=y_min + 0.8 * (y_max - y_min),
                    s=f'mean = {mean: 0.1f}')
        # draw median
        median_line_kws = None
        if isinstance(median_line, bool):
            if median_line is True:
                median_line_kws = {
                    'color': 'blue',
                    'linestyle': ':'
                }
        else:
            median_line_kws = dict(median_line)
            if 'line_style' in median_line_kws:
                median_line_kws['linestyle'] = median_line_kws.pop('line_style')
            if 'line_width' in median_line_kws:
                median_line_kws['linewidth'] = median_line_kws.pop('line_width')
        if median_line_kws is not None:
            median = self._data.median()
            ax.axvline(x=median, **median_line_kws)
            y_min, y_max = ax.get_ylim()
            ax.text(x=median, y=y_min + 0.7 * (y_max - y_min),
                    s=f'median = {median: 0.1f}')
        return ax

    def plot_hist(
            self,
            color: Color = 'k',
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot a histogram of the distribution.

        :param color: Color of the bars.
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        histplot(
            data=self._data,
            color=color,
            ax=ax
        )
        return ax

    def split_by(
            self,
            categorical: CategoricalDistribution
    ):
        """
        Split into a RatioSeries on different values of the given categorical
        distribution.

        :param categorical: Distribution to split on
        """
        ratios_dict = {}
        for category in categorical.categories:
            ratios_dict[category] = self.filter_to(categorical.keep(category))
        from probability.distributions.data.ratio_series import RatioSeries
        ratio_series_data = Series(ratios_dict, name=self.name)
        ratio_series_data.index.name = categorical.name
        return RatioSeries(ratio_series_data)

    def probably_less_than(self, other: 'Ratio') -> float:
        """
        Find the approximate probability that self < other, assuming an
        underlying Normal data generating distribution.
        """
        m_self = self._data.mean()
        s_self = self._data.std()
        m_other = other._data.mean()
        s_other = other._data.std()
        m_diff = m_self - m_other
        s_diff = (s_self ** 2 + s_other ** 2) ** 0.5
        diff = Normal(mu=m_diff, sigma=s_diff)
        return diff < 0

    def probably_greater_than(self, other: 'Ratio') -> float:
        """
        Find the approximate probability that self > other, assuming an
        underlying Normal data generating distribution.
        """
        m_self = self._data.mean()
        s_self = self._data.std()
        m_other = other._data.mean()
        s_other = other._data.std()
        m_diff = m_self - m_other
        s_diff = (s_self ** 2 + s_other ** 2) ** 0.5
        diff = Normal(mu=m_diff, sigma=s_diff)
        return diff > 0

    def plot_conditional_dist_densities(
            self,
            categorical: CategoricalDistribution,
            fit_dist: Type[RVContinuous1dMixin],
            hdi: float = 0.95,
            width: float = 0.8,
            num_segments: int = 100,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            color_median: Optional[Color] = None,
            edge_color: Optional[Color] = None,
            ax: Optional[Axes] = None
    ) -> Axes:
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
        :param ax: Optional matplotlib Axes to plot on.
        """
        ax = ax or new_axes()
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
            add_v_density(
                ax=ax,
                x=c + 1,
                y_to_z=cat_dist.pdf().at(
                    linspace(y_min, y_max, num_segments + 1)),
                color=color, color_min=color_min, edge_color=edge_color,
                width=width
            )
            # plot descriptive statistics lines
            if color_mean is not None:
                mean = cat_ratio_data.mean()
                ax.plot([c + 0.55, c + 1.45], [mean, mean], color=color_mean)
            if color_median is not None:
                median = cat_ratio_data.median()
                ax.plot([c + 0.55, c + 1.45], [median, median],
                        color=color_median)
        # labels
        ax.set_title(f'{hdi: .0%} HDIs of p({self.name}|{categorical.name})')
        ax.set_xlabel(categorical.name)
        ax.set_ylabel(self.name)
        # axes
        ax.set_xlim(0, n_cats + 1)
        yy_range = yy_max - yy_min
        ax.set_ylim(yy_min - yy_range * 0.05, yy_max + yy_range * 0.05)
        ax.set_xticks(range(1, n_cats + 1))
        ax.set_xticklabels(cats)

        return ax


    def residualize(
            self,
            X: Union[DataFrame, Series, ndarray, List[Any], Any],
            keep_mean: bool = True
    ) -> 'Ratio':
        """
        Frisch-Waugh-Lovell (FWL) partialling-out / residualization of this Ratio outcome on covariates X.

        Regresses this Ratio distribution on X (with intercept) and computes the residuals:
            tilde_y = y - X beta_hat
        If keep_mean is True (default), adds back the sample mean y_bar so the adjusted
        distribution remains on the original unit scale:
            y_adjusted = tilde_y + y_bar

        :param X: Confounder(s) or covariate(s) to partial out (Series, DataFrame, or array).
        :param keep_mean: If True, shift residuals back by original mean (preserving unit scale).
        :return: A new Ratio distribution containing the adjusted / residualized values.
        """
        from probability.models.linear_regression import LinearRegression

        model = LinearRegression.fit(
            X=X,
            y=self._data,
            fit_intercept=True,
            cov_type='nonrobust'
        )
        resids = model.residuals
        if keep_mean:
            adjusted = resids + float(self._data.loc[resids.index].mean())
        else:
            adjusted = resids

        adjusted.name = f"{self.name or 'ratio'}_residualized"
        return Ratio(adjusted)

    def regress(
            self,
            X: Union[DataFrame, Series, ndarray, List[Any], Any],
            cov_type: Optional[str] = None,
            cluster: Optional[Any] = None,
            fit_intercept: bool = True
    ) -> 'LinearRegression':
        """
        Fit an OLS regression of this Ratio outcome on covariates X with robust standard errors.

        :param X: Predictors / covariates (DataFrame, Series, array, or distribution objects).
        :param cov_type: Covariance estimator ('HC3' default, 'HC2', 'HC1', 'HC0', 'nonrobust', 'cluster').
        :param cluster: Cluster identifiers for cluster-robust standard errors.
        :param fit_intercept: Whether to fit an intercept term (default True).
        :return: Fitted LinearRegression instance.
        """
        from probability.models.linear_regression import LinearRegression

        return LinearRegression.fit(
            X=X,
            y=self,
            cov_type=cov_type,
            cluster=cluster,
            fit_intercept=fit_intercept
        )

    def __repr__(self):

        return (
            f'{self.name}: Ratio['
            f'min={self.min()}, '
            f'max={self.max()}, '
            f'mean={self.mean()}'
            f']'
        )

