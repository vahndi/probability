from typing import Union, Optional, Iterable

from matplotlib.axes import Axes
from numpy import clip
from pandas import Series, DataFrame, concat
from seaborn import kdeplot

from probability.distributions.data.ratio import Ratio
from probability.plotting import (
    Color,
    add_rectangle,
    format_axis_integer,
    new_axes,
)
from probability.distributions.continuous.normal_series import NormalSeries
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesStdMixin, DataSeriesVarMixin, DataSeriesSemMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin, DataSeriesTrendMixin


class RatioSeries(
    DataSeriesMixin,
    DataSeriesTrendMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesStdMixin,
    DataSeriesVarMixin,
    DataSeriesSemMixin,
    object
):
    """
    Series or dict of Ratio distributions.
    """

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

    def to_normal(self) -> NormalSeries:
        """
        Infer a Series of Normal distributions, one for each Ratio distribution.
        """
        return NormalSeries(
            Series({
                key: self._data[key].to_normal()
                for key in self.keys()
            }, name=self.name)
        )

    def histograms(
            self,
            bins: Union[int, float, Iterable[float]],
            min_pct: float = 0.0,
            max_pct: float = 1.0
    ):
        """
        Return a Series mapping distribution names to histograms.

        :param bins: int number of bins, float bin spacing, or sequence of bin
                     edges.
        :param min_pct: Lowest percentile of data to use for the histogram.
        :param max_pct: Highest percentile of data to use for the histogram.
        """
        dist: Ratio
        return Series({
            ix: dist.histogram(bins=bins, min_pct=min_pct, max_pct=max_pct)
            for ix, dist in self._data.items()
        })

    def unsplit(self) -> Ratio:
        """
        Return a Ratio with all the data for each Ratio in the
        RatioSeries concatenated.
        """
        return Ratio(
            data=concat([
                self._data[key].data for key in self.keys()
            ])
        )

    def _sample_series(self, unsplit_dist: Ratio, n_x: Series) -> 'RatioSeries':
        y_random = unsplit_dist.data.sample(frac=1)
        index = []
        for k in self.keys():
            index.extend([k] * n_x[k])
        y_random.index = index
        s_x = {
            k: Ratio(Series(data=y_random.loc[y_random.index == k]))
            for k in self.keys()
        }
        return RatioSeries(s_x)

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
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot a density bar for each Ratio distribution.

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
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        dist: Ratio
        # calculate histograms, counts and bin limits
        hists = [
            dist.histogram(
                bins=float(bin_spacing),
                min_pct=min_pct, max_pct=max_pct
            )
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
                add_rectangle(
                    ax=ax,
                    width=width, height=max_val - min_val,
                    x_center=x, y_center=(min_val + max_val) / 2,
                    color=color, line_width=0,
                    alpha=float(alpha)
                )
            # add density edges
            if edge_color is not None:
                low_bin = hist.index.get_level_values('min')[0]
                high_bin = hist.index.get_level_values('max')[-1]
                add_rectangle(
                    ax=ax,
                    width=width, height=high_bin - low_bin,
                    x_center=x, y_center=(low_bin + high_bin) / 2,
                    fill=False, color=None, edge_color=edge_color
                )
        # format axes
        ax.set_xlim(-1, len(self._data))
        ax.set_ylim(lowest_bin - bin_spacing, highest_bin + bin_spacing)
        ax.set_xticks(range(len(self._data)))
        ax.set_xticklabels(self._data.index.to_list())

        format_axis_integer(ax.yaxis)
        ax.set_title(
            f'Distribution of '
            f'p({self.data.name}|{self.data.index.name})'
        )
        ax.set_xlabel(self.data.index.name)
        ax.set_ylabel(self.data.name)

        return ax

    def plot_density_distributions(
            self,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot the distribution of probability density for each Ratio
        distribution.

        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        name: str
        dist: Ratio
        for name, dist in self._data.items():
            kdeplot(
                x=dist.data,
                label=name, ax=ax
            )
        ax.legend()
        return ax
