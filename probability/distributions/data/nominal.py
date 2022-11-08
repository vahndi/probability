from itertools import combinations
from typing import List, Union, Optional, TYPE_CHECKING, Tuple

from numpy import nan
from pandas import Series

from probability.distributions import Beta
from probability.distributions.continuous.beta_series import BetaSeries
from probability.distributions.data.boolean import Boolean
from probability.distributions.data.ordinal import Ordinal
from probability.distributions.mixins.data.data_discrete_categorical_mixin import \
    DataDiscreteCategoricalMixin
from probability.distributions.mixins.data.data_probability_table_mixin import \
    DataProbabilityTableMixin
from probability.distributions.mixins.data.data_information_mixin import \
    DataInformationMixin
from probability.distributions.mixins.data.data_categories_mixin import \
    DataCategoriesMixin
from probability.distributions.mixins.data.data_distribution_mixin import \
    DataDistributionMixin
from probability.distributions.mixins.data.data_aggregate_mixins import \
    DataModeMixin

if TYPE_CHECKING:
    from probability.distributions.data.nominal_series import NominalSeries


class Nominal(
    DataDistributionMixin,
    DataCategoriesMixin,
    DataDiscreteCategoricalMixin,
    DataModeMixin,
    DataProbabilityTableMixin,
    DataInformationMixin,
    object
):

    _ordered = False  # used for categorical mixin methods

    def __init__(self, data: Series):
        """
        Create a new Ordinal distribution.

        :param data: Categorical pandas Series.
        """
        data = data.dropna()
        self._data: Series = data
        self._categories: List[str] = data.cat.categories.to_list()

    def as_ordinal(
            self,
            ordered_categories: List[str],
    ) -> Ordinal:
        """
        Convert to an Ordinal distribution with the given categories.
        Any categories not given will be dropped.

        :param ordered_categories: List of ordered categories.
        """
        data = self.drop([
            c for c in self.categories
            if c not in ordered_categories
        ]).data
        data = data.cat.set_categories(ordered_categories, ordered=True)
        return Ordinal(data=data)

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

    def to_beta_series(self) -> BetaSeries:

        return BetaSeries(
            data=Series(data={k: Beta(alpha=v, beta=len(self) - v)
                              for k, v in self.counts().items()},
                        name=self.name),
        )

    def split_by(
            self,
            categorical: Union[DataCategoriesMixin, DataDistributionMixin]
    ) -> 'NominalSeries':
        """
        Split into an OrdinalSeries on different values of the given categorical
        distribution.

        :param categorical: Distribution to split on
        """
        nominals_dict = {}
        for category in categorical.categories:
            nominals_dict[category] = self.filter_to(categorical.keep(category))
        from probability.distributions.data.nominal_series import NominalSeries
        nominal_series_data = Series(nominals_dict, name=self.name)
        nominal_series_data.index.name = categorical.name
        return NominalSeries(nominal_series_data)

    def category_combinations(self, k: int) -> List[Tuple[str, ...]]:
        """
        Return combinations of category names.

        :param k: Number of names in each combination
        """
        return list(combinations(self.categories, k))

    def __repr__(self):

        cat_counts = self._data.value_counts().reindex(self._categories)
        str_cat_counts = ', '.join([
            f'"{cat}": {count}'
            for cat, count in cat_counts.items()
        ])
        return f'{self.name}: Nominal[{str_cat_counts}]'
