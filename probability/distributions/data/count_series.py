from typing import Optional

from numpy.random import choice
from pandas import Series, concat

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade, set_alpha
from probability.distributions import Count
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesModeMixin, DataSeriesMedianMixin
from probability.distributions.mixins.data.data_series_category_mixins import \
    DataSeriesCountsMixin, DataSeriesPMFsMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin, DataSeriesTrendMixin


class CountSeries(
    DataSeriesMixin,
    DataSeriesTrendMixin,
    DataSeriesCountsMixin,
    DataSeriesPMFsMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesModeMixin,
    DataSeriesMedianMixin,
    object
):
    """
    Series or dict of Count distributions.
    """

    def unsplit(self) -> Count:
        """
        Return a Count with all the data for each Count in the
        CountSeries concatenated.
        """
        return Count(
            data=concat([
                self._data[key].data for key in self.keys()
            ])
        )

    def _sample_series(self, unsplit_dist: Count, n_x: Series) -> 'CountSeries':
        p_y = unsplit_dist.pmf()
        y = p_y.index.to_list()
        s_x = {
            k: Count(Series(data=choice(a=y, size=n_x[k], p=p_y)))
            for k in self.keys()
        }
        return CountSeries(s_x)

    def plot_bars(
            self,
            conditional: bool = False,
            max_pct: Optional[float] = None,
            width: float = 0.8,
            height: float = 0.8,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            alpha_min: float = 0.0,
            edge_color: Optional[Color] = 'grey',
            axf: Optional[AxesFormatter] = None
    ):
        """
        Plot a set of bars for each Series, shaded by the count at each discrete
        value.

        :param conditional: Whether the shading for each Series should be
                            independent of all the others.
        :param max_pct: Highest percentile of each Series to show a bar for.
        :param width: Width of each set of bars.
        :param height: Height of each bar, centered about the count value.
        :param color: Color of each bar.
        :param color_min: Optional different color for the sparsest bars.
        :param alpha_min: Alpha value for the sparsest bars.
        :param edge_color: Optional edge color of each bar.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        dist: Count
        # find highest count of all Series
        max_count = max(
            counts.max() for counts in self.counts()
        )
        # find highest value for y-axis scaling
        if max_pct is None:
            # find highest value of all Series
            max_val = self.max().max()
        else:
            max_val = 0
        # plot each Series
        for x, (ix, dist) in enumerate(self._data.items()):
            dist_items = dist.counts()
            if max_pct:
                max_dist_val = dist.data.quantile(max_pct)
                dist_items = dist_items.loc[dist_items.index <= max_dist_val]
                max_val = max(max_val, max_dist_val)
            if conditional:
                max_count = dist_items.max()
            for dist_value, dist_count in dist_items.items():
                coords = dict(
                    width=width, height=height,
                    x_center=(1 + x),
                    y_center=dist_value
                )
                if color_min is not None:
                    from_color = set_alpha(color_min, alpha_min)
                else:
                    from_color = set_alpha(color, alpha_min)
                axf.add_rectangle(
                    **coords,
                    color=cross_fade(
                        from_color=from_color,
                        to_color=color,
                        amount=dist_count / max_count
                    )
                )
                if edge_color is not None:
                    axf.add_rectangle(
                        **coords,
                        fill=False, color=None, edge_color=edge_color
                    )
        axf.set_x_lim(0.5, len(self._data) + 0.5)
        axf.set_y_lim(self.min().min() - 1, max_val + 1)
        axf.x_ticks.set_locations(range(1, len(self._data) + 1))
        axf.x_ticks.set_labels(self._data.index.to_list())
        axf.y_axis.set_format_integer()

        return axf


if __name__ == '__main__':

    from probability.distributions import Poisson
    dists = {}
    for i in range(5, 10):
        dists[i] = Count(Poisson(i).rvs(10_000))
    cs = CountSeries(Series(dists))
    cs.plot_bars(color='red').show()
    cs.plot_bars(color='red', conditional=True).show()
    cs.plot_bars(color='red', color_min='yellow', conditional=True).show()
    cs.plot_bars(color='red', color_min='yellow',
                 alpha_min=0.5, conditional=True).show()
