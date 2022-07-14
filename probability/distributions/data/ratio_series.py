from typing import Union, Dict, Any, Optional, Iterable

from numpy import clip
from pandas import Series, DataFrame, concat
from seaborn import kdeplot
from tqdm import tqdm

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from probability.distributions import Ratio
from probability.distributions.continuous.normal_series import NormalSeries
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesStdMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin


class RatioSeries(
    DataSeriesMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesStdMixin,
    object
):
    """
    Series or dict of Ratio distributions.
    """
    def __init__(self, data: Union[Series, Dict[Any, Ratio]]):
        """
        Create a new RatioSeries.

        :param data: Series of Ratio distributions.
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

    def unsplit(self) -> Ratio:
        """
        Return an Ordinal with all the data for each Ordinal in the
        OrdinalSeries concatenated.
        """
        return Ratio(
            data=concat([
                self._data[key].data for key in self.keys()
            ])
        )

    def p_increasing(
            self,
            n_iter: int = 1_000
    ) -> float:
        """
        Return the probability that the Ratios increase as the key increases.
        """
        ref_probs = self.increase_probs()
        ref_sum = ref_probs.sum()
        ref_prod = ref_probs.product()
        # calculate y
        y = self.unsplit()
        # calculate n[X]
        n_x = self.lens()
        # repeat
        results = []
        for _ in tqdm(range(n_iter)):
            # create sampled distribution for each x
            s_x = {}
            y_random = y.data.sample(frac=1)
            index = []
            # create new index on the random samples
            for k in self.keys():
                index.extend([k] * n_x[k])
            y_random.index = index
            # create a new ratio series using the new index
            for k in self.keys():
                s_x[k] = Ratio(Series(
                    data=y_random.loc[y_random.index == k]
                ))
            rat_test = RatioSeries(s_x)
            # find probability that sampled distribution is increasing,
            # and by how much
            test_probs = rat_test.increase_probs()
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
        Return the probability that the Ordinals decrease as the key increases.
        """
        ref_probs = self.increase_probs()
        ref_sum = ref_probs.sum()
        ref_prod = ref_probs.product()
        # calculate y
        y = self.unsplit()
        # calculate n[X]
        n_x = self.lens()
        # repeat
        results = []
        for _ in tqdm(range(n_iter)):
            # create sampled distribution for each x
            s_x = {}
            y_random = y.data.sample(frac=1)
            index = []
            # create new index on the random samples
            for k in self.keys():
                index.extend([k] * n_x[k])
            y_random.index = index
            # create a new ratio series using the new index
            for k in self.keys():
                s_x[k] = Ratio(Series(
                    data=y_random.loc[y_random.index == k]
                ))
            rat_test = RatioSeries(s_x)
            # find probability that sampled distribution is increasing,
            # and by how much
            test_probs = rat_test.increase_probs()
            test_sum = test_probs.sum()
            test_prod = test_probs.prod()
            # if result is more extreme than observed, record a 1
            if ref_sum < test_sum and ref_prod < test_prod:
                results.append(1)
            else:
                results.append(0)
        return Series(results).mean()

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
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
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
        axf.set_x_lim(-1, len(self._data))
        axf.set_y_lim(lowest_bin - bin_spacing, highest_bin + bin_spacing)
        axf.x_ticks.set_locations(range(len(self._data)))
        axf.x_ticks.set_labels(self._data.index.to_list())
        axf.y_axis.set_format_integer()
        axf.set_text(
            title=f'Distribution of '
                  f'p({self.data.name}|{self.data.index.name})',
            x_label=self.data.index.name,
            y_label=self.data.name
        )

        return axf

    def plot_density_distributions(
            self,
            axf: Optional[AxesFormatter] = None
    ):
        """
        Plot the distribution of probability density for each Ratio
        distribution.

        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        name: str
        dist: Ratio
        for name, dist in self._data.items():
            kdeplot(
                x=dist.data,
                label=name, ax=axf.axes
            )
        axf.axes.legend()
        return axf
