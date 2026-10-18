from typing import List, Any, Optional, Tuple

from matplotlib.axes import Axes
from pandas import Series, concat

from probability.distributions.mixins.data.data_counts_mixin import \
    DataCountsMixin
from probability.plotting import (
    Color,
    FONT_SIZE,
    FontSize,
    add_text,
    format_as_integer,
    format_as_percent,
    format_axis_integer,
    format_axis_percent,
    new_axes,
)


class DataDiscreteCategoricalMixin(
    DataCountsMixin,
    object
):

    name: str
    _data: Series
    _categories: List[Any]

    def plot_bars(
            self,
            color: Color = 'k',
            pct_font_size: FontSize = FONT_SIZE.medium,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot a bar plot of the counts of each category.

        :param color: Color of the bars.
        :param pct_font_size: Font size for percentage labels.
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        counts = self.counts()
        pmf = self.pmf()
        counts.plot.bar(ax=ax, color=color)
        percents = 100 * pmf
        add_text(
            ax=ax,
            x=range(len(counts)), y=counts,
            text=percents.map(lambda p: f'{p: .1f}%'),
            h_align='center', v_align='bottom',
            font_size=pct_font_size
        )
        format_axis_integer(ax.yaxis)
        return ax

    def plot_comparison_bars(
            self,
            other: 'DataDiscreteCategoricalMixin',
            absolute: bool = False,
            color: Tuple[Color, Color] = ('C0', 'C1'),
            width: float = 0.5,
            label_pcts: bool = True,
            label_counts: bool = False,
            label_size: Optional[FontSize] = FONT_SIZE.medium,
            ax: Optional[Axes] = None
    ) -> Axes:
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
        :param ax: Optional matplotlib Axes instance.
        """
        # validation
        if self.name == other.name:
            raise ValueError(
                'Distributions must have different names in order to compare.')
        self_cats, other_cats = set(self._categories), set(other._categories)
        if self_cats != other_cats:
            str_warning = f'WARNING: Distributions contain different categories'
            unique_1 = self_cats.difference(other_cats)
            unique_2 = other_cats.difference(self_cats)
            if unique_1:
                str_warning += f'\nOnly in {self.name}: {", ".join(unique_1)}'
            if unique_2:
                str_warning += f'\nOnly in {other.name}: {", ".join(unique_2)}'
            print(str_warning)
        # get data
        self_counts = self._data.value_counts().reindex(self._categories)
        other_counts = other._data.value_counts().reindex(self._categories)
        count_data = concat([self_counts, other_counts], axis=1)
        pct_data = count_data / count_data.sum()
        # plot bars
        ax = ax or new_axes()
        if absolute:
            plot_data = count_data
        else:
            plot_data = pct_data
        plot_data.plot.bar(ax=ax, color=color, width=width)
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
                add_text(
                    ax, label_x, label_y, text,
                    font_size=label_size,
                    h_align='center', v_align='bottom'
                )
        # format y-axis
        if not absolute:
            format_axis_percent(ax.yaxis)
        else:
            format_axis_integer(ax.yaxis)

        return ax
