from typing import Union, Dict, Any, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
from numpy import linspace, inf
from pandas import Series, DataFrame

from probability.distributions import Boolean, BetaBinomialConjugate
from probability.plotting import Color, add_v_density, new_axes
from probability.distributions.continuous.beta_series import BetaSeries
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesModeMixin, DataSeriesMedianMixin, DataSeriesStdMixin, \
    DataSeriesVarMixin, DataSeriesSemMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin
from probability.distributions.protocols import CategoricalDistribution


if TYPE_CHECKING:
    from probability.distributions.data.boolean_frame import BooleanFrame


class BooleanSeries(
    DataSeriesMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesModeMixin,
    DataSeriesMedianMixin,
    DataSeriesStdMixin,
    DataSeriesVarMixin,
    DataSeriesSemMixin,
    object
):
    """
    Series or dict of Boolean distributions.
    """

    def split_by(
            self,
            categorical: CategoricalDistribution
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
            ax: Optional[Axes] = None
    ) -> Axes:
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
        :param ax: Optional matplotlib Axes to plot on.
        """
        ax = ax or new_axes()

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
                mean = cat_data.mean()
                ax.plot([c + 0.55, c + 1.45], [mean, mean], color=color_mean)
        # labels
        ax.set_title(
            f'{hdi: .0%} HDIs of $p(' +
            r'p_{' + self.name + r'}' +
            f'|{self.index.name})$'
        )
        ax.set_xlabel(self.index.name)
        ax.set_ylabel(r'$p_{' + self.name + r'}$')
        # axes
        ax.set_xlim(0, n_cats + 1)
        yy_range = yy_max - yy_min
        ax.set_ylim(yy_min - yy_range * 0.05, yy_max + yy_range * 0.05)
        ax.set_yticks(linspace(0, 1, 11))
        ax.set_xticks(range(1, n_cats + 1))
        ax.set_xticklabels(cats)

        return ax

