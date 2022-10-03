from random import seed
from typing import Union, Tuple, List, Optional, TYPE_CHECKING

from numpy import linspace, inf
from pandas import Series

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from probability.distributions import Skellam
from probability.distributions.conjugate.gamma_poisson_conjugate import \
    GammaPoissonConjugate
from probability.distributions.data.ordinal import Ordinal
from probability.distributions.mixins.data.data_aggregate_mixins import \
    DataMinMixin, DataMaxMixin, DataMeanMixin, DataMedianMixin, DataStdMixin, \
    DataModeMixin, DataVarMixin
from probability.distributions.mixins.data.data_categories_mixin import \
    DataCategoriesMixin
from probability.distributions.mixins.data.data_discrete_numeric_mixin import \
    DataDiscreteNumericMixin
from probability.distributions.mixins.data.data_distribution_mixin import \
    DataDistributionMixin
from probability.distributions.mixins.data.data_information_mixin import \
    DataInformationMixin
from probability.distributions.mixins.data.data_numeric_comparison_mixin \
    import DataNumericComparisonMixin
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON

if TYPE_CHECKING:
    from probability.distributions.data.count_series import CountSeries


class Count(
    DataDistributionMixin,
    DataNumericComparisonMixin,
    DataDiscreteNumericMixin,
    DataInformationMixin,
    DataMinMixin,
    DataMaxMixin,
    DataMeanMixin,
    DataMedianMixin,
    DataStdMixin,
    DataVarMixin,
    DataModeMixin,
    object
):
    def __init__(self, data: Series):
        """
        Create a new Count distribution.

        :param data: pandas Series.
        """
        data = data.dropna()
        self._data: Series = data.astype(int)
        self._categories = list(range(self._data.min(), self._data.max() + 1))

    @property
    def categories(self) -> List[int]:

        return self._categories

    @property
    def num_categories(self) -> int:

        return len(self._categories)

    def as_ordinal(
            self,
            values: Optional[List[Union[int, Tuple[int, int]]]] = None
    ):
        """
        Convert to an Ordinal variable.

        :param values: List of values or value ranges to map to categories.
                       e.g. [1, 2, (3, 5), (6, None)] will give an
                       Ordinal with categories ['1', '2', '3-5', '6+']
        """
        mapping = {}
        new_categories = []
        if values is None:
            values = self._categories
        for value in values:
            if isinstance(value, int):
                category_name = str(value)
                new_categories.append(category_name)
                mapping[value] = category_name
            else:
                min_val = int(value[0])
                max_val = int(value[1]) if value[1] is not None else None
                if max_val is not None:
                    category_name = f'{min_val}-{max_val}'
                else:
                    category_name = f'{min_val}+'
                    max_val = int(self._data.max())
                new_categories.append(category_name)
                for val in range(min_val, max_val + 1):
                    mapping[val] = category_name
        new_data = self._data.map(mapping).astype('category')
        new_data = new_data.cat.set_categories(new_categories, ordered=True)
        return Ordinal(data=new_data)

    def plot_conditional_prob_densities(
            self,
            categorical: Union[DataDistributionMixin, DataCategoriesMixin],
            hdi: float = 0.95,
            width: float = 0.8,
            num_segments: int = 100,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            color_median: Optional[Color] = None,
            edge_color: Optional[Color] = None,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot conditional probability densities of the data, split by the
        categories of an Ordinal or Nominal distribution.

        :param categorical: Nominal or Ordinal distribution.
        :param hdi: Highest Density Interval width for each distribution.
        :param width: Width of each density bar.
        :param num_segments: Number of segments to plot per density.
        :param color: Color for the densest part of each distribution.
        :param color_min: Color for the sparsest part of each distribution,
                          if different to color.
        :param color_mean: Color for mean data markers.
        :param color_median: Color for median data markers.
        :param edge_color: Optional color for the edge of each density bar.
        :param axf: Optional AxesFormatter to plot on.
        """
        axf = axf or AxesFormatter()
        cats = categorical.categories
        n_cats = len(cats)
        yy_min, yy_max = inf, -inf
        # filter categorical data
        shared_ix = list(
            set(self._data.index).intersection(categorical.data.index)
        )
        cat_data = categorical.data.loc[shared_ix]
        ratio_data = self._data.loc[shared_ix]
        for c, category in enumerate(categorical.categories):
            cat_ratio_data = ratio_data.loc[cat_data == category]
            if len(cat_ratio_data) == 0:
                continue
            # fit distribution and find limits for HDI
            cat_dist = GammaPoissonConjugate.infer_posterior(
                data=cat_ratio_data)
            # cat_dist = fit_dist.fit(data=cat_ratio_data)
            y_min, y_max = cat_dist.hdi(hdi)
            yy_min, yy_max = min(y_min, yy_min), max(y_max, yy_max)
            # plot density
            axf.add_v_density(
                x=c + 1,
                y_to_z=cat_dist.pdf().at(
                    linspace(y_min, y_max, num_segments + 1)),
                color=color, color_min=color_min, edge_color=edge_color,
                width=width
            )
            # plot descriptive statistics lines
            if color_mean is not None:
                mean = cat_ratio_data.mean()
                axf.add_line(x=[c + 0.55, c + 1.45], y=[mean, mean],
                             color=color_mean)
            if color_median is not None:
                median = cat_ratio_data.median()
                axf.add_line(x=[c + 0.55, c + 1.45], y=[median, median],
                             color='g')
        # labels
        axf.set_text(
            title=f'{hdi: .0%} HDIs of $p(' +
                  r'\lambda_{' + self.name + r'}' +
                  f'|{categorical.name})$',
            x_label=categorical.name,
            y_label=r'$\lambda_{' + self.name + r'}$'
        )
        # axes
        axf.set_x_lim(0, n_cats + 1)
        yy_range = yy_max - yy_min
        axf.set_y_lim(yy_min - yy_range * 0.05, yy_max + yy_range * 0.05)
        axf.x_ticks.set_locations(range(1, n_cats + 1)).set_labels(cats)

        return axf

    def split_by(
            self,
            categorical: Union[DataCategoriesMixin, DataDistributionMixin]
    ) -> 'CountSeries':
        """
        Split into a CountSeries on different values of the given categorical
        distribution.

        :param categorical: Distribution to split on
        """
        counts_dict = {}
        for category in categorical.categories:
            counts_dict[category] = self.filter_to(categorical.keep(category))
        from probability.distributions.data.count_series import CountSeries
        count_series_data = Series(counts_dict, name=self.name)
        count_series_data.index.name = categorical.name
        return CountSeries(count_series_data)

    def _check_can_compare(self, other: 'Count'):

        if not isinstance(other, Count):
            raise TypeError('Can only compare a Count with another Count')

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

    def _comparison_samples(
            self, other: 'Count',
            num_samples: Optional[int] = None
    ) -> Tuple[Series, Series]:
        """
        Method to make it faster and more accurate to implement  <= and >=.
        """
        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON
        self_samples = self.rvs(num_samples)
        other_samples = other.rvs(num_samples)
        return self_samples, other_samples

    def prob_equal_to(self, other: 'Count',
                      num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self = other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values == other_values).mean()

    def __eq__(self, other: 'Count') -> float:
        """
        Return the probability that self = other. based on sampling.
        """
        return self.prob_equal_to(other)

    def prob_not_equal_to(self, other: 'Count',
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self != other. based on sampling.
        """
        return 1 - self.prob_equal_to(other, num_samples)

    def __ne__(self, other: 'Count') -> float:
        """
        Return the probability that self != other. based on sampling.
        """
        return self.prob_not_equal_to(other)

    def prob_less_than(self, other: 'Count',
                       num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self < other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values < other_values).mean()

    def __lt__(self, other: 'Count') -> float:
        """
        Return the probability that self < other. based on sampling.
        """
        return self.prob_less_than(other)

    def prob_greater_than(self, other: 'Count',
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self > other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values > other_values).mean()

    def __gt__(self, other: 'Count') -> float:
        """
        Return the probability that self > other. based on sampling.
        """
        return self.prob_greater_than(other)

    def __le__(self, other: 'Count') -> float:
        """
        Return the probability that self <= other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values <= other_values).mean()

    def __ge__(self, other: 'Count') -> float:
        """
        Return the probability that self >= other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values >= other_values).mean()

    def probably_less_than(self, other: 'Count') -> float:
        """
        Find the approximate probability that self < other, assuming an
        underlying Poisson data generating distribution.
        """
        m_self = self._data.mean()
        m_other = other._data.mean()
        diff = Skellam(m_self, m_other)
        return diff < 0

    def probably_greater_than(self, other: 'Count') -> float:
        """
        Find the approximate probability that self > other, assuming an
        underlying Poisson data generating distribution.
        """
        m_self = self._data.mean()
        m_other = other._data.mean()
        diff = Skellam(m_self, m_other)
        return diff > 0

    def __repr__(self):

        return (
            f'{self.name}: Count['
            f'min={self._data.min()}, '
            f'max={self._data.max()}, '
            f'mean={self._data.mean()}'
            f']'
        )
