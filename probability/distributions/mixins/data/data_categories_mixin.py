from collections import OrderedDict
from typing import TypeVar, List, Union, Optional, TYPE_CHECKING

from numpy import nan
from pandas import Series

from probability.custom_types.external_custom_types import FloatArray1d
from probability.distributions.conjugate.dirichlet_multinomial_conjugate \
    import DirichletMultinomialConjugate
from probability.distributions.continuous.beta_series import BetaSeries

if TYPE_CHECKING:
    from probability.distributions.data.boolean import Boolean

DCM = TypeVar('DCM', bound='DataCategoriesMixin')


class DataCategoriesMixin(object):

    _categories: List[Union[bool, str]]
    _data: Series
    _ordered: bool
    name: str

    @property
    def categories(self) -> list:
        """
        Return the names of the data categories.
        """
        return self._categories

    @property
    def num_categories(self) -> int:
        """
        Return the number of data categories.
        """
        return len(self._categories)

    def drop(self: DCM, categories: Union[str, List[str]]) -> DCM:
        """
        Drop one or more categories from the underlying data.

        :param categories: Categories to drop.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[~self._data.isin(categories)]
        new_cats = [cat for cat in self._categories
                    if cat not in categories]
        data = data.cat.set_categories(
            new_categories=new_cats,
            ordered=self._ordered
        )
        return type(self)(data=data)

    def keep(self: DCM, categories: Union[str, List[str]]) -> DCM:
        """
        Drop all the categories from the data not in the one(s) given.

        :param categories: Categories to keep.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[self._data.isin(categories)]
        new_cats = [cat for cat in self._categories if cat in categories]
        data = data.cat.set_categories(
            new_categories=new_cats,
            ordered=self._ordered
        )
        return type(self)(data=data)

    def rename_categories(
            self: DCM,
            new_categories: Union[list, dict]
    ) -> DCM:
        """
        Return a new instance with its categories renamed.
        """
        if (
                isinstance(new_categories, list) or
                (isinstance(new_categories, dict) and
                 len(new_categories.values()) ==
                 len(set(new_categories.values())))
        ):
            # one to one
            return type(self)(
                data=self._data.cat.rename_categories(new_categories)
            )
        else:
            # many to one
            data = self._data.astype(object).replace(new_categories).astype('category')
            categories = list(OrderedDict.fromkeys(new_categories.values()))
            data = data.cat.set_categories(
                new_categories=categories,
                ordered=self._ordered
            )
            return type(self)(data=data)

    def reverse_categories(self: DCM) -> DCM:
        """
        Return a new instance with its category order reversed. Leaves the data
        untouched.
        """
        data = self._data.cat.set_categories(
            new_categories=self.categories[::-1],
            ordered=self._ordered
        )
        return type(self)(data=data)

    def as_boolean(
            self,
            false: Union[str, List[str]],
            true: Union[str, List[str]],
            empty: Optional[Union[str, List[str]]] = None
    ) -> 'Boolean':
        """
        Convert to a Boolean distribution.

        :param false: Categories to map to False.
        :param true: Categories to map to True.
        :param empty: Categories to map to nan.
        """
        from probability.distributions.data.boolean import Boolean
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
        mapping = {}
        for f in false:
            mapping[f] = False
        for t in true:
            mapping[t] = True
        for e in empty:
            mapping[e] = nan
        data = self._data.astype(object).map(mapping).dropna().astype(bool)
        return Boolean(data)

    def pmf_betas(
            self,
            alpha: Optional[Union[FloatArray1d, dict, float]] = None
    ) -> BetaSeries:
        """
        Return a BetaSeries with the probability of each category as a
        Beta distribution.

        :param alpha: Value(s) for the α hyper-parameter of the prior Dirichlet
                      distribution. Defaults to Uniform distribution,
        """
        dirichlet = DirichletMultinomialConjugate.infer_posterior(
            data=self._data, alpha=alpha
        )
        return BetaSeries(Series({
            name: dirichlet[name]
            for name in dirichlet.names
        }))

