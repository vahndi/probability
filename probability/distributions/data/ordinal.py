from typing import List, Optional, Tuple, Union

from numpy import nan
from numpy.random import seed
from pandas import Series, concat

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
from mpl_format.utils.number_utils import format_as_percent
from probability.distributions.data.boolean import Boolean
from probability.distributions.data.interval import Interval
from probability.distributions.mixins.data_mixins import \
    DataDistributionMixin, DataProbabilityTableMixin, DataMinMixin, DataMaxMixin, \
    DataInformationMixin, DataCategoriesMixin, DataDiscreteMixin
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class Ordinal(
    DataDistributionMixin,
    DataCategoriesMixin,
    DataDiscreteMixin,
    DataMinMixin,
    DataMaxMixin,
    DataProbabilityTableMixin,
    DataInformationMixin,
    object
):
    """
    Ordinal data is a categorical, statistical data type where the variables
    have natural, ordered categories and the distances between the categories
    are not known. These data exist on an ordinal scale, one of four levels of
    measurement described by S. S. Stevens in 1946. The ordinal scale is
    distinguished from the nominal scale by having a ranking. It also differs
    from the interval scale and ratio scale by not having category widths that
    represent equal increments of the underlying attribute.

    https://en.wikipedia.org/wiki/Ordinal_data
    """

    _ordered = True  # used for categorical mixin methods

    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: Categorical pandas Series.
        """
        data = data.dropna()
        self._data: Series = data
        self._categories: List[str] = data.cat.categories.to_list()
        self._name_to_val = {
            category: ix
            for ix, category in enumerate(self._categories)
        }
        self._val_to_name = {
            ix: category
            for ix, category in enumerate(self._categories)
        }
        self._data_vals: Series = self._data.replace(self._name_to_val)

    def rvs(self, num_samples: int,
            random_state: Optional[int] = None) -> Series:
        """
        Sample `num_samples` random values from the distribution.
        """
        if random_state is not None:
            seed(random_state)
        return self._data.sample(
            n=num_samples, replace=True
        ).reset_index(drop=True)

    def rvs_values(self, num_samples: int,
                   random_state: Optional[int] = None) -> Series:
        """
        Sample `num_samples` random values from the distribution.
        """
        return self._data_vals.sample(
            n=num_samples, replace=True,
            random_state=random_state
        ).reset_index(drop=True)

    def correlation(self, other: 'Ordinal') -> float:
        """
        Calculate the Spearman rank correlation coefficient with another
        Ordinal.
        """
        combined = concat([
            self._data_vals, other._data_vals
        ], axis=1)
        return combined.corr(method='spearman').iloc[0, 1]

    def median_value(self) -> int:

        return self._data_vals.median()

    def median(self) -> str:

        return self._val_to_name[self.median_value()]

    def mode(self) -> Union[str, List[str]]:

        mode = self._data_vals.mode()
        if len(mode) > 1:
            return mode.to_list()
        else:
            return mode[0]

    def as_interval(self, offset: int = 0) -> Interval:
        """
        Convert to an Interval distribution.

        :param offset: Optional number to add to each interval value e.g. to
                       get from a 5-point (0:4) scale to a (-2:2) scale use
                       offset=-2.
        """
        return Interval(data=self._data_vals + offset)

    def as_boolean(
            self,
            false: Union[str, List[str]],
            true: Union[str, List[str]],
            empty: Optional[Union[str, List[str]]] = None
    ) -> Boolean:
        """
        Convert to a Boolean distribution.

        :param false: Categories to map to False.
        :param true: Categories to map to True.
        :param empty: Categories to map to nan.
        """
        if not isinstance(false, list):
            false = [false]
        if not isinstance(true, list):
            true = [true]
        if empty is None:
            empty = []
        elif not isinstance(empty, list):
            empty = [empty]
        if not set(true + false + empty) == set(self.categories):
            raise ValueError('Must provide all categories in the distribution')
        data = self._data.copy()
        for f in false:
            data = data.replace(f, False)
        for t in true:
            data = data.replace(t, True)
        for e in empty:
            data = data.replace(e, nan)
        data = data.dropna()
        return Boolean(data)

    def plot_conditional_dist_densities(
            self,
            categorical: Union[DataDistributionMixin, DataCategoriesMixin],
            width: float = 0.8,
            heights: float = 0.9,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            color_median: Optional[Color] = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot conditional probability densities of the data, split by the
        categories of an Ordinal or Nominal distribution.

        :param categorical: Nominal or Ordinal distribution.
        :param width: Width of each density bar.
        :param heights: Height of each density bar.
        :param color: Color for the densest part of each distribution.
        :param color_min: Color for the sparsest part of each distribution,
                          if different to color.
        :param color_mean: Color for mean data markers.
        :param color_median: Color for median data markers.
        :param axf: Optional AxesFormatter to plot on.
        """
        axf = axf or AxesFormatter()
        cats = categorical.categories
        n_cats = len(cats)
        # filter categorical data
        shared_ix = list(
            set(self._data.index).intersection(categorical.data.index)
        )
        cat_data = categorical.data.loc[shared_ix]
        ordinal_data = self._data.loc[shared_ix]
        max_cat_sum = max([
            ordinal_data.loc[cat_data == category].value_counts().sum()
            for category in categorical.categories
        ])
        max_pct = max([
            ordinal_data.loc[cat_data == category].value_counts().max() /
            ordinal_data.loc[cat_data == category].value_counts().sum()
            for category in categorical.categories
        ])
        for c, category in enumerate(categorical.categories):
            cat_ratio_data = ordinal_data.loc[cat_data == category]
            value_counts = cat_ratio_data.value_counts().reindex(
                self.categories).fillna(0)
            cat_sum = value_counts.sum()
            for i, (item, count) in enumerate(value_counts.items()):
                pct = count / cat_sum
                if color_min is not None:
                    rect_color = cross_fade(color_min, color, pct / max_pct)
                else:
                    rect_color = color
                bar_width = width * cat_sum / max_cat_sum
                x_center = 1 + c
                y_center = 1 + i
                axf.add_rectangle(
                    width=bar_width, height=heights,
                    x_left=x_center - bar_width / 2,
                    y_bottom=y_center - heights / 2,
                    color=rect_color,
                    alpha=pct / max_pct
                )
                axf.add_text(x=x_center, y=y_center,
                             text=format_as_percent(pct, 1),
                             h_align='center', v_align='center',
                             bbox_edge_color='k', bbox_fill=True,
                             bbox_face_color='white')
            if len(cat_ratio_data) == 0:
                continue
            # plot descriptive statistics lines
            if color_mean is not None:
                interval = 1 + cat_ratio_data.cat.codes
                mean = interval.mean()
                axf.add_line(x=[c + 0.55, c + 1.45], y=[mean, mean],
                             color=color_mean)
            if color_median is not None:
                interval = 1 + cat_ratio_data.cat.codes
                median = interval.median()
                axf.add_line(x=[c + 0.55, c + 1.45], y=[median, median],
                             color='g')
        # # labels
        axf.set_text(
            title=f'Distributions of p({self.name}|{categorical.name})',
            x_label=categorical.name,
            y_label=self.name
        )
        # # axes
        axf.set_x_lim(0.5, n_cats + 0.5)
        # yy_range = yy_max - yy_min
        axf.set_y_lim(0, len(self._categories) + 1)
        axf.x_ticks.set_locations(range(1, n_cats + 1)).set_labels(cats)
        axf.y_ticks.set_locations(
            range(1, len(self._categories) + 1)).set_labels(self._categories)

        return axf

    def _check_can_compare(self, other: 'Ordinal'):

        if not isinstance(other, Ordinal):
            raise TypeError('Can only compare an Ordinal with another Ordinal')
        if not self._categories == other._categories:
            raise ValueError('Both Ordinals must have the same categories.')

    def _comparison_samples(
            self, other: 'Ordinal',
            num_samples: Optional[int] = None
    ) -> Tuple[Series, Series]:

        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON
        self_samples = self.rvs_values(num_samples)
        other_samples = other.rvs_values(num_samples)
        return self_samples, other_samples

    def prob_equal_to(self, other: 'Ordinal',
                      num_samples: int = NUM_SAMPLES_COMPARISON) -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values == other_values).mean()

    def __eq__(self, other: 'Ordinal') -> float:

        return self.prob_equal_to(other)

    def prob_not_equal_to(self, other: 'Ordinal',
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:

        return 1 - self.prob_equal_to(other, num_samples)

    def __ne__(self, other: 'Ordinal') -> float:

        return self.prob_not_equal_to(other)

    def prob_less_than(self, other: 'Ordinal',
                       num_samples: int = NUM_SAMPLES_COMPARISON) -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values < other_values).mean()

    def __lt__(self, other: 'Ordinal') -> float:

        return self.prob_less_than(other)

    def probably_less_than(self, other: 'Ordinal') -> float:
        """
        Find the approximate probability that self > other.

        Assigns a proportion of the samples that are equal to greater than
        according to the relative proportion of greater than to greater than
        plus less than.
        """
        gt = self > other
        lt = self < other
        eq = self == other
        return lt + (lt / (lt + gt)) * eq

    def prob_greater_than(self, other: 'Ordinal',
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values > other_values).mean()

    def __gt__(self, other: 'Ordinal') -> float:

        return self.prob_greater_than(other)

    def probably_greater_than(
            self, other: 'Ordinal',
            num_samples: int = NUM_SAMPLES_COMPARISON
    ) -> float:
        """
        Find the approximate probability that self > other.

        Assigns a proportion of the samples that are equal to greater than
        according to the relative proportion of greater than to greater than
        plus less than.
        """
        gt = self.prob_greater_than(other, num_samples)
        lt = self.prob_less_than(other, num_samples)
        eq = self.prob_equal_to(other, num_samples)
        return gt + (gt / (gt + lt)) * eq

    def __le__(self, other: 'Ordinal') -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values <= other_values).mean()

    def __ge__(self, other: 'Ordinal') -> float:

        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values >= other_values).mean()

    def __repr__(self):

        cat_counts = self._data.value_counts().reindex(self._categories)
        str_cat_counts = ', '.join([
            f'"{cat}": {count}'
            for cat, count in cat_counts.items()
        ])
        return f'{self.name}: Ordinal[{str_cat_counts}]'
