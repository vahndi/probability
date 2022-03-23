from typing import List, Any, Optional, Tuple

from pandas import Series, concat

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color, FontSize
from mpl_format.enums import FONT_SIZE
from mpl_format.utils.number_utils import format_as_percent, format_as_integer
from probability.distributions.mixins.data.data_counts_mixin import \
    DataCountsMixin


class DataDiscreteNumericMixin(
    DataCountsMixin,
    object
):

    name: str
    _data: Series
    _categories: List[Any]

    def plot_bars(
            self,
            color: Color = 'k',
            pct_font_size: int = FONT_SIZE.medium,
            max_pct: Optional[float] = None,
            max_value: Optional[int] = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot a bar plot of the counts of each category.

        :param color: Color of the bars.
        :param pct_font_size: Font size for percentage labels.
        :param max_pct: Highest percentile value of the data to plot a count of.
                        Useful for long-tail distributions.
        :param max_value: Highest value of the data to plot a count of.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        counts = self.counts()
        pmf = self.pmf()
        if max_pct is not None:
            max_count = self._data.quantile(max_pct)
            counts = counts.loc[counts.index <= max_count]
            pmf = pmf.loc[counts.index]
        if max_value is not None:
            counts = counts.loc[counts.index <= max_value]
            pmf = pmf.loc[counts.index]
        counts.plot.bar(ax=axf.axes, color=color)
        percents = 100 * pmf
        axf.add_text(
            x=range(len(counts)), y=counts,
            text=percents.map(lambda p: f'{p: .1f}%'),
            h_align='center', v_align='bottom',
            font_size=pct_font_size
        )
        axf.y_axis.set_format_integer()
        return axf

    def plot_comparison_bars(
            self,
            other: 'DataDiscreteNumericMixin',
            absolute: bool = False,
            color: Tuple[Color, Color] = ('C0', 'C1'),
            width: float = 0.5,
            label_pcts: bool = True,
            label_counts: bool = False,
            label_size: Optional[FontSize] = FONT_SIZE.medium,
            max_pct: Optional[float] = None,
            max_value: Optional[int] = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot a comparison of the 2 ordinals, with outlines around bars that
        are significantly higher or lower than others.

        :param other: Another Ordinal with the same categories.
        :param absolute: Whether to plot bar heights as absolute values or
                        percentages.
        :param color: Color for each set of bars.
        :param width: Total width of each pair of bars.
        :param label_pcts: Whether to add percentage labels.
        :param label_counts: Whether to add count labels.
        :param label_size: Font size for bar labels.
        :param max_pct: Highest percentile value of the data to plot a count of.
                        Useful for long-tail distributions.
        :param max_value: Highest value of the data to plot a count of.
        :param axf: Optional AxesFormatter instance.
        """
        # validation
        if self.name == other.name:
            raise ValueError(
                'Distributions must have different names in order to compare.')
        self_cats, other_cats = set(self._categories), set(other._categories)
        categories = sorted(self_cats.union(other_cats))
        # get data
        self_counts = self._data.value_counts().reindex(categories)
        other_counts = other._data.value_counts().reindex(categories)
        count_data = concat([self_counts, other_counts], axis=1)
        pct_data = count_data / count_data.sum()
        if max_pct is not None:
            max_count = max(
                self._data.quantile(max_pct),
                other._data.quantile(max_pct)
            )
            self_counts = self_counts.loc[self_counts.index <= max_count]
            other_counts = other_counts.loc[other_counts.index <= max_count]
        if max_value is not None:
            self_counts = self_counts.loc[self_counts.index <= max_value]
            other_counts = other_counts.loc[other_counts.index <= max_value]
        count_data = concat([self_counts, other_counts], axis=1)
        pct_data = pct_data.loc[count_data.index]
        # plot bars
        axf = axf or AxesFormatter()
        if absolute:
            plot_data = count_data
        else:
            plot_data = pct_data
        plot_data.plot.bar(ax=axf.axes, color=color, width=width)
        # add labels
        for o, ordinal in enumerate(plot_data.columns):
            for i, ix in enumerate(plot_data[ordinal].index):
                if not label_pcts and not label_counts:
                    continue
                label_x = i - width / 4 + o * width / 2
                label_y = plot_data.loc[ix, ordinal]
                texts = []
                if label_pcts:
                    texts.append(format_as_percent(pct_data.loc[ix, ordinal]))
                if label_counts:
                    texts.append(format_as_integer(count_data.loc[ix, ordinal]))
                text = ' | '.join(texts)
                axf.add_text(label_x, label_y, text,
                             font_size=label_size,
                             h_align='center', v_align='bottom')
        # format y-axis
        if not absolute:
            axf.y_axis.set_format_percent()
        else:
            axf.y_axis.set_format_integer()

        return axf
