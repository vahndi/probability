from typing import Union, Dict, Any, Optional, TYPE_CHECKING

from numpy import linspace, inf
from pandas import Series, DataFrame

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from probability.distributions import Boolean, BetaBinomialConjugate
from probability.distributions.continuous.beta_series import BetaSeries
from probability.distributions.mixins.data.data_categories_mixin import \
    DataCategoriesMixin
from probability.distributions.mixins.data.data_distribution_mixin import \
    DataDistributionMixin
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesModeMixin, DataSeriesMedianMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


if TYPE_CHECKING:
    from probability.distributions.data.boolean_frame import BooleanFrame


class BooleanSeries(
    DataSeriesMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesModeMixin,
    DataSeriesMedianMixin,
    object
):
    """
    Series or dict of Boolean distributions.
    """
    def __init__(self, data: Union[Series, Dict[Any, Boolean]]):
        """
        Create a new BooleanSeries.

        :param data: Series of Boolean distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data

    def split_by(
            self,
            categorical: Union[DataCategoriesMixin, DataDistributionMixin]
    ) -> 'BooleanFrame':
        """
        Split into a BooleanSeries on different values of the given categorical
        distribution.

        :param categorical: Distribution to split on
        """
        bools = []
        for self_cat in self.keys():
            self_cat_dist: Boolean = self._data[self_cat]
            for other_cat in categorical.categories:
                self_other_dist = self_cat_dist.filter_to(
                    categorical.keep(other_cat)
                )
                bools.append({
                    self.name: self_cat,
                    categorical.name: other_cat,
                    'value': self_other_dist
                })
        bools_frame = DataFrame(bools).pivot(
            index=categorical.name,
            columns=self.name,
            values='value'
        ).reindex(index=categorical.categories, columns=self.keys())
        from probability.distributions.data.boolean_frame import BooleanFrame
        return BooleanFrame(bools_frame)

    def to_beta_series(self) -> BetaSeries:
        """
        Convert to a Beta distribution.
        """
        return BetaSeries(data=self._data.map(lambda b: b.to_beta()))

    def plot_prob_densities(
            self,
            hdi: float = 0.95,
            width: float = 0.8,
            num_segments: int = 100,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            edge_color: Optional[Color] = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot conditional probability densities of the data, split by the
        categories of an Ordinal or Nominal distribution.

        :param hdi: Highest Density Interval width for each distribution.
        :param width: Width of each density bar.
        :param num_segments: Number of segments to plot per density.
        :param color: Color for the densest part of each distribution.
        :param color_min: Color for the sparsest part of each distribution,
                          if different to color.
        :param color_mean: Color for mean data markers.
        :param edge_color: Optional color for the edge of each density bar.
        :param axf: Optional AxesFormatter to plot on.
        """
        axf = axf or AxesFormatter()

        # cats = categorical.categories
        cats = list(self.keys())
        n_cats = len(self._data)
        yy_min, yy_max = inf, -inf
        for c, category in enumerate(self.keys()):
            # fit distribution and find limits for HDI
            cat_data: Series = self._data[category].data
            cat_dist = BetaBinomialConjugate.infer_posterior(cat_data)
            # cat_dist = Beta.fit(data=cat_ratio_data)
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
                mean = cat_data.mean()
                axf.add_line(x=[c + 0.55, c + 1.45], y=[mean, mean],
                             color=color_mean)
        # labels
        axf.set_text(
            title=f'{hdi: .0%} HDIs of $p(' +
                  r'p_{' + self.name + r'}' +
                  f'|{self.index.name})$',
            x_label=self.index.name,
            y_label=r'$p_{' + self.name + r'}$'
        )
        # axes
        axf.set_x_lim(0, n_cats + 1)
        yy_range = yy_max - yy_min
        axf.set_y_lim(yy_min - yy_range * 0.05, yy_max + yy_range * 0.05)
        axf.y_ticks.set_locations(linspace(0, 1, 11))
        axf.x_ticks.set_locations(range(1, n_cats + 1)).set_labels(cats)

        return axf
