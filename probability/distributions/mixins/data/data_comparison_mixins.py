from itertools import product
from typing import TypeVar, Dict

from pandas import Series, DataFrame

from probability.distributions.mixins.data.data_categories_mixin import \
    DataCategoriesMixin

DCDM = TypeVar('DCDM')


class DataCohensDMixin(object):

    _data: Series

    def cohens_d(self: DCDM, other: DCDM) -> float:
        """
        Calculate the Cohen's d standardized difference of means between self
        and other.

        https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
        """
        n1, n2 = len(self), len(other)
        v1, v2 = self._data.var(), other._data.var()
        x1, x2 = self._data.mean(), other._data.mean()
        # pooled standard deviation
        s = (
            ((n1 - 1) * v1 + (n2 - 1) * v2) /
            (n1 + n2 - 2)
        ) ** 0.5
        return (x1 - x2) / s

    def conditional_cohens_d(
            self: DCDM,
            categorical: DataCategoriesMixin
    ) -> DataFrame:
        """
        Return a matrix of the Cohen's d of the Ratio conditioned on each
        category compared to it conditioned on each other category.
        Index represents control and columns represent treatment.

        :param categorical: Categorical data distribution to condition on.

        https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
        """
        dists: Dict[str, DCDM] = {
            category: self.filter_to(
                categorical.keep(category)
            ).rename(category)
            for category in categorical.categories
        }
        v: Dict[str, float] = {
            category: dist.var()
            for category, dist in dists.items()
        }
        n: Dict[str, int] = {
            category: len(dist)
            for category, dist in dists.items()
        }
        x: Dict[str, float] = {
            category: dist.mean()
            for category, dist in dists.items()
        }
        results = []
        for control, treatment in product(dists.keys(), dists.keys()):
            n1, n2 = n[control], n[treatment]
            v1, v2 = v[control], v[treatment]
            x1, x2 = x[control], x[treatment]
            s = (
                    ((n1 - 1) * v1 + (n2 - 1) * v2) /
                    (n1 + n2 - 2)
            ) ** 0.5
            results.append({
                'control': control,
                'treatment': treatment,
                'd': (x2 - x1) / s
            })
        return DataFrame(results).pivot(
            index='control', columns='treatment', values='d'
        ).reindex(dists.keys(), axis=0).reindex(dists.keys(), axis=1)
