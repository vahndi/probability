from typing import Optional, Union, Dict, Any

from pandas import Series

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade, set_alpha
from probability.distributions import Count


class CountSeries(object):
    """
    Series of Count distributions.
    """
    def __init__(self, data: Union[Series, Dict[Any, Count]]):
        """
        Create a new CountSeries.

        :param data: Series of Count distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data

    def min(self) -> Series:

        return Series({
            ix: self._data[ix].min()
            for ix in self._data.index
        })

    def pmfs(self) -> Series:

        return Series({
            ix: self._data[ix].pmf()
            for ix in self._data.index
        })

    def counts(self) -> Series:

        return Series({
            ix: self._data[ix].counts()
            for ix in self._data.index
        })

    def max(self) -> Series:

        return Series({
            ix: self._data[ix].max()
            for ix in self._data.index
        })

    def plot_bars(
            self,
            conditional: bool = False,
            max_pct: Optional[float] = None,
            width: float = 0.8,
            height: float = 0.8,
            color: Color = 'k',
            edge_color: Optional[Color] = 'grey',
            axf: Optional[AxesFormatter] = None
    ):

        axf = axf or AxesFormatter()
        dist: Count
        max_count = max(
            counts.max() for counts in self.counts()
        )
        if max_pct is None:
            max_val = self.max().max()
        else:
            max_val = 0
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
                axf.add_rectangle(
                    **coords,
                    color=cross_fade(
                        from_color=set_alpha(color, 0),
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
