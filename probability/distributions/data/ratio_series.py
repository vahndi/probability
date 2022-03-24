from typing import Union, Dict, Any, Optional

from numpy import clip
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

    def plot_density_bars(
            self,
            bin_spacing: float,
            conditional: bool = True,
            width: float = 0.8,
            min_pct: float = 0.0,
            max_pct: float = 1.0,
            min_alpha: float = 0.0,
            color: Color = 'k',
            edge_color: Optional[Color] = 'grey',
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        :param bin_spacing: Spacing between each bin used to calculate
                            histogram of each Ratio distribution.
        :param conditional: Whether to set opacity relative to the highest bin
                            count of each Series (True) or all Series (False).
        :param width: Width of each bar,
        :param min_pct: Minimum quantile to plot from 0.0 to 1.0.
        :param max_pct: Maximum quantile to plot from 0.0 to 1.0.
        :param min_alpha: Alpha value for the opacity of bins with 0 count.
        :param color: Color for each density bar.
        :param edge_color: Color for edge of each bar. Set to None to omit
                           edges.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        dist: Ratio
        # calculate histograms, counts and bin limits
        hists = [
            dist.histogram(bins=bin_spacing, min_pct=min_pct, max_pct=max_pct)
            for _, dist in self._data.items()
        ]
        lowest_bin = min([
            hist.index.get_level_values('min')[0] for hist in hists])
        highest_bin = max([
            hist.index.get_level_values('max')[-1] for hist in hists])
        max_counts = [hist.max() for hist in hists]
        max_max_count = max(max_counts)
        for x, hist, max_count in zip(
                range(len(self._data)), hists, max_counts
        ):
            max_count = hist.max()
            for (min_val, max_val), count in hist.items():
                # add segment interior
                if conditional is True:
                    alpha = clip(
                        min_alpha + (1 - min_alpha) * count / max_count,
                        min_alpha, 1.0
                    )
                else:
                    alpha = clip(
                        min_alpha + (1 - min_alpha) * count / max_max_count,
                        min_alpha, 1.0
                    )
                axf.add_rectangle(
                    width=width, height=max_val - min_val,
                    x_center=x, y_center=(min_val + max_val) / 2,
                    color=color, line_width=0,
                    alpha=alpha
                )
            # add density edges
            if edge_color is not None:
                low_bin = hist.index.get_level_values('min')[0]
                high_bin = hist.index.get_level_values('max')[-1]
                axf.add_rectangle(
                    width=width, height=high_bin - low_bin,
                    x_center=x, y_center=(low_bin + high_bin) / 2,
                    fill=False, color=None, edge_color=edge_color
                )
        # format axes
        axf.set_x_lim(-1, len(self._data) + 0.5)
        axf.set_y_lim(lowest_bin - bin_spacing, highest_bin + bin_spacing)
        axf.x_ticks.set_locations(range(len(self._data)))
        axf.x_ticks.set_labels(self._data.index.to_list())
        axf.y_axis.set_format_integer()

        return axf
