from typing import Optional, Union, Dict, Any

from numpy.random import choice
from pandas import Series, DataFrame, concat
from tqdm import tqdm

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade, set_alpha
from mpl_format.utils.number_utils import format_as_percent
from probability.distributions import Count
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesModeMixin, DataSeriesMedianMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


class CountSeries(
    DataSeriesMixin,
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
    def __init__(self, data: Union[Series, Dict[Any, Count]]):
        """
        Create a new CountSeries.

        :param data: Series of Count distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data

    def pmfs(self) -> Series:
        """
        Return a Series mapping each distribution name to its pmf.
        """
        return Series({
            ix: self._data[ix].pmf()
            for ix in self._data.index
        })

    def counts(self) -> Series:
        """
        Return a Series mapping each distribution name to its counts.
        """
        return Series({
            ix: self._data[ix].counts()
            for ix in self._data.index
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

    def p_increasing(
            self,
            n_iter: int = 1_000
    ) -> float:
        """
        Return the probability that the Counts increase as the key increases.
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
                s_x[k] = Count(Series(
                    data=choice(a=y, size=n_x[k], p=p_y),
                ))
            count_test = CountSeries(s_x)
            # find probability that sampled distribution is increasing,
            # and by how much
            test_probs = count_test.increase_probs()
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
        Return the probability that the Counts decrease as the key increases.
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
                s_x[k] = Count(Series(
                    data=choice(a=y, size=n_x[k], p=p_y),
                ))
            count_test = CountSeries(s_x)
            # find probability that sampled distribution is increasing,
            # and by how much
            test_probs = count_test.increase_probs()
            test_sum = test_probs.sum()
            test_prod = test_probs.prod()
            # if result is more extreme than observed, record a 1
            if ref_sum < test_sum and ref_prod < test_prod:
                results.append(1)
            else:
                results.append(0)
        return Series(results).mean()

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
            pct_labels: bool = True,
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
        :param pct_labels: Whether to add percentage labels to each bar.
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
            dist_items_orig = dist_items
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
                if pct_labels:
                    pct = dist_count / dist_items_orig.sum()
                    axf.add_text(x=coords['x_center'], y=coords['y_center'],
                                 text=format_as_percent(pct, 1),
                                 h_align='center', v_align='center',
                                 bbox_edge_color='k', bbox_fill=True,
                                 bbox_face_color='white')
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
