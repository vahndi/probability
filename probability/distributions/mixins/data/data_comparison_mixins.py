from itertools import product
from typing import TypeVar, Dict, Optional, Tuple

from pandas import Series, DataFrame

from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from probability.distributions.protocols import (
    CategoricalDistribution, NumericDataDistribution
)

DCDM = TypeVar('DCDM', bound=NumericDataDistribution)
DPCM = TypeVar('DPCM', bound='DataProbComparisonMixin')


class DataProbComparisonMixin(object):

    def _check_can_compare(self: DPCM, other: DPCM):

        if not isinstance(other, type(self)):
            cls_name = type(self).__name__
            article = 'an' if cls_name[0].lower() in 'aeiou' else 'a'
            raise TypeError(
                f'Can only compare {article} {cls_name} with another {cls_name}'
            )

    def _comparison_samples(
            self: DPCM, other: DPCM,
            num_samples: Optional[int] = None
    ) -> Tuple[Series, Series]:
        """
        Method to make it faster and more accurate to implement <= and >=.
        """
        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON
        self_samples = self.rvs(num_samples)
        other_samples = other.rvs(num_samples)
        return self_samples, other_samples

    def prob_equal_to(self: DPCM, other: DPCM,
                      num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self = other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values == other_values).mean()

    def __eq__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self = other. based on sampling.
        """
        return self.prob_equal_to(other)

    def prob_not_equal_to(self: DPCM, other: DPCM,
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self != other. based on sampling.
        """
        return 1 - self.prob_equal_to(other, num_samples)

    def __ne__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self != other. based on sampling.
        """
        return self.prob_not_equal_to(other)

    def prob_less_than(self: DPCM, other: DPCM,
                       num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self < other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values < other_values).mean()

    def __lt__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self < other. based on sampling.
        """
        return self.prob_less_than(other)

    def prob_greater_than(self: DPCM, other: DPCM,
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self > other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values > other_values).mean()

    def __gt__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self > other. based on sampling.
        """
        return self.prob_greater_than(other)

    def __le__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self <= other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values <= other_values).mean()

    def __ge__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self >= other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values >= other_values).mean()


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
            categorical: CategoricalDistribution
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

