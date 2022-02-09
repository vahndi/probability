from typing import Union, List, Optional, Iterable, Tuple, TypeVar

from numpy import log
from pandas import Series, DataFrame, concat
from scipy.stats import entropy

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color, FontSize
from mpl_format.enums import FONT_SIZE
from mpl_format.utils.number_utils import format_as_percent, format_as_integer


T = TypeVar('T', bound='DataDistributionMixin')


class DataDistributionMixin(object):

    _data: Series

    @property
    def data(self) -> Series:
        """
        Return the underlying data used to construct the Distribution.
        """
        return self._data

    @property
    def name(self):
        """
        Return the name of the Series of data.
        """
        return self._data.name

    def rename(self: T, name: str) -> T:
        """
        Rename the Series of data.
        """
        return type(self)(data=self._data.rename(name))

    def filter_to(self: T, other: 'DataDistributionMixin') -> T:
        """
        Filter the data to the common indices with the other distribution.
        """
        shared_ix = list(set(self._data.index).intersection(other.data.index))
        data = self._data.loc[shared_ix]
        return type(self)(data=data)


class DataCategoriesMixin(object):

    _categories: List[Union[bool, str]]
    _data: Series
    name: str

    @property
    def categories(self) -> list:

        return self._categories

    def plot_bars(
            self,
            color: Color = 'k',
            pct_font_size: int = FONT_SIZE.medium,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:

        axf = axf or AxesFormatter()
        counts = self._data.value_counts().reindex(self._categories)
        counts.plot.bar(ax=axf.axes, color=color)
        percents = 100 * counts / len(self._data.dropna())
        axf.add_text(
            x=range(len(counts)), y=counts,
            text=percents.map(lambda p: f'{p: .1f}%'),
            h_align='center', v_align='bottom',
            font_size=pct_font_size
        )
        axf.y_axis.set_format_integer()
        return axf

    def plot_comparison_bars(
            self, other: 'DataCategoriesMixin',
            absolute: bool = False,
            color: Tuple[Color] = ('C0', 'C1'),
            width: float = 0.5,
            sig_colors: Optional[Tuple[Color]] = ('green', 'red'),
            label_pcts: bool = True,
            label_counts: bool = False,
            label_size: Optional[FontSize] = FONT_SIZE.medium,
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
        :param sig_colors: Colors for significantly higher and lower bar borders
        :param label_pcts: Whether to add percentage labels.
        :param label_counts: Whether to add count labels.
        :param label_size: Font size for bar labels.
        :param axf: Optional AxesFormatter instance.
        """
        # validation
        self_cats, other_cats = set(self._categories), set(other._categories)
        if self_cats != other_cats:
            str_warning = f'WARNING: Ordinals contain different categories'
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
        # add colors for significance
        if sig_colors is not None:
            pass
        # format y-axis
        if not absolute:
            axf.y_axis.set_format_percent()
        else:
            axf.y_axis.set_format_integer()

        return axf


class DataMinMixin(object):

    _data: Series

    def min(self) -> Union[float, int, str]:
        """
        Return the smallest value in the data.
        """
        return self._data.min()


class DataMaxMixin(object):

    _data: Series

    def max(self) -> Union[float, int, str]:
        """
        Return the largest value in the data.
        """
        return self._data.max()


class DataMeanMixin(object):

    _data: Series

    def mean(self) -> float:
        """
        Return the mean value of the data.
        """
        return self._data.mean()


class DataMedianMixin(object):

    _data: Series

    def median(self) -> Union[int, float]:
        """
        Return the median value of the data.
        """
        return self._data.median()


class DataStdMixin(object):

    _data: Series

    def std(self) -> float:
        """
        Return the median value of the data.
        """
        return self._data.std()


class DataModeMixin(object):

    _data: Series

    def mode(self) -> Union[int, float, str,
                            List[int], List[float], List[str]]:
        """
        Return the most frequently occurring value(s) in the data.
        """
        mode = self._data.mode()
        if len(mode) > 1:
            return mode.to_list()
        else:
            return mode[0]


class DataCPTMixin(object):

    _data: Series

    def cpt(self, condition: 'DataCPTMixin') -> DataFrame:
        """
        Return the conditional probability of each category given different
        values of condition.
        """
        self_name = self._data.name
        other_name = condition._data.name
        data = concat([condition._data, self._data], axis=1)
        joint = data[[other_name, self_name]].value_counts()
        marginal = data[other_name].value_counts()
        marginal.index.name = other_name
        return (joint / marginal).unstack()


class DataInformationMixin(object):

    _data: Series

    def _calc_frame(self, other: 'DataInformationMixin') -> DataFrame:
        """
        Calculate the joint and marginal probability distributions of self and
        other and return as a DataFrame.
        """
        x_counts = self._data.value_counts().rename_axis('x')
        y_counts = other._data.value_counts().rename_axis('y')
        xy_counts = concat([
            self._data.rename('x'),
            other._data.rename('y')
        ], axis=1).value_counts()
        p_x = x_counts / x_counts.sum()
        p_y = y_counts / y_counts.sum()
        p_xy = xy_counts / xy_counts.sum()
        calc = p_xy.rename('p(x,y)').to_frame()
        calc['p(x)'] = p_x.reindex(
            calc.index.get_level_values('x')).to_list()
        calc['p(y)'] = p_y.reindex(
            calc.index.get_level_values('y')).to_list()
        return calc

    def entropy(self) -> float:
        """
        Return the entropy of the distribution (self-information).
        """
        return entropy(self._data.value_counts())

    def mutual_information(self, other: 'DataInformationMixin') -> float:
        """
        In probability theory and information theory, the mutual information
        (MI) of two random variables is a measure of the mutual dependence
        between the two variables. More specifically, it quantifies the
        "amount of information" (in units such as shannons (bits), nats or
        hartleys) obtained about one random variable by observing the other
        random variable. The concept of mutual information is intimately linked
        to that of entropy of a random variable, a fundamental notion in
        information theory that quantifies the expected "amount of information"
        held in a random variable.

        https://en.wikipedia.org/wiki/Mutual_information
        """
        calc = self._calc_frame(other)
        calc['I(x,y)'] = calc['p(x,y)'] * (
                calc['p(x,y)'] / (calc['p(x)'] * calc['p(y)'])
        ).map(log)
        return calc['I(x,y)'].sum()

    def conditional_entropy(self, other: 'DataInformationMixin') -> float:
        """
        In information theory, the conditional entropy quantifies the amount of
        information needed to describe the outcome of a random variable Y given
        that the value of another random variable X is known. Here, information
        is measured in shannons, nats, or hartleys. The entropy of Y conditioned
        on X is written as H(Y|X)}.

        https://en.wikipedia.org/wiki/Conditional_entropy
        """
        calc = self._calc_frame(other)
        calc['H(x|y)'] = calc['p(x,y)'] * (
                calc['p(x,y)'] / calc['p(x)']
        ).map(log)
        return -calc['H(x|y)'].sum()

    def joint_entropy(self, other: 'DataInformationMixin') -> float:
        """
        In information theory, joint entropy is a measure of the uncertainty
        associated with a set of variables.

        https://en.wikipedia.org/wiki/Joint_entropy
        """
        calc = self._calc_frame(other)
        calc['H(x,y)'] = calc['p(x,y)'] * calc['p(x,y)'].map(log)
        return -calc['H(x,y)'].sum()

    def relative_mutual_information(self, other: 'DataInformationMixin'):
        """
        Return the proportion of entropy in self explained by observing
        other.
        """
        return self.mutual_information(other) / self.entropy()
