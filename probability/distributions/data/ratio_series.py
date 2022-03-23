from typing import Union, Dict, Any, Optional, Sized, List
from math import floor, ceil, inf
from numpy import arange, histogram, clip
from pandas import Series, DataFrame

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from probability.distributions import Ratio


class RatioSeries(object):
    """
    Series of Ratio distributions.
    """
    def __init__(self, data: Union[Series, Dict[Any, Ratio]]):
        """
        Create a new CountSeries.

        :param data: Series of Count distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data

    @staticmethod
    def from_data_frame(
            data: DataFrame,
            ratio: str,
            split_by: str
    ) -> 'RatioSeries':
        """
        Create a RatioSeries containing a column of ratio data, and a column of
        discrete data to split by.

        :param data: DataFrame to use.
        :param ratio: Name of the column with ratio data.
        :param split_by: Name of the column with discrete data.
        """
        split_ratios = {}
        for unique in sorted(data[split_by].unique()):
            split_ratios[unique] = Ratio(data.loc[
                data[split_by] == unique,
                ratio
            ].rename(f'{ratio}|{split_by}={unique}'))
        return RatioSeries(split_ratios)

    def plot_bars(
            self,
            bin_spacing: float,
            width: float = 0.8,
            min_pct: float = 0.0,
            max_pct: float = 1.0,
            min_alpha: float = 0.0,
            color: Color = 'k',
            edge_color: Optional[Color] = 'grey',
            axf: Optional[AxesFormatter] = None
    ):

        axf = axf or AxesFormatter()
        dist: Ratio
        highest_bin = -inf
        lowest_bin = inf
        for x, (ix, dist) in enumerate(self._data.items()):
            low_bin = (
                    bin_spacing * floor(dist.data.quantile(min_pct) /
                                        bin_spacing)
            )
            high_bin = (
                    bin_spacing * ceil(dist.data.quantile(max_pct) /
                                       bin_spacing)
            )
            if high_bin > highest_bin:
                highest_bin = high_bin
            if low_bin < lowest_bin:
                lowest_bin = low_bin
            bins = arange(low_bin, high_bin + bin_spacing, bin_spacing)
            hist, _ = histogram(a=dist.data, bins=bins)
            max_count = hist.max()
            for b in range(len(hist)):
                coords = dict(
                    width=width, height=bins[b + 1] - bins[b],
                    x_center=x, y_center=(bins[b + 1] + bins[b]) / 2
                )
                axf.add_rectangle(
                    **coords,
                    color=color,
                    alpha=clip(
                        min_alpha + (1 - min_alpha) * hist[b] / max_count,
                        min_alpha, 1.0
                    ),
                    line_width=0
                )
                if edge_color is not None:
                    axf.add_rectangle(
                        **coords,
                        fill=False, color=None, edge_color=edge_color
                    )
        axf.set_x_lim(0.5, len(self._data) + 0.5)
        axf.set_y_lim(lowest_bin - bin_spacing, highest_bin + bin_spacing)
        axf.x_ticks.set_locations(range(1, len(self._data) + 1))
        axf.x_ticks.set_labels(self._data.index.to_list())
        axf.y_axis.set_format_integer()

        return axf
