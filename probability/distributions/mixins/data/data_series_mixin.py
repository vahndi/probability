from typing import List, Hashable, Union, Dict, Any

from pandas import Series, Index, DataFrame
from tqdm import tqdm

from probability.distributions.protocols import SupportsTrendSeries


class DataSeriesMixin(object):

    _data: Series

    def __init__(self, data: Union[Series, Dict[Any, Any]]):
        """
        Create a new DataSeries.

        :param data: Series or dict of distributions.
        """
        if isinstance(data, dict):
            data = Series(data)
        self._data: Series = data

    @property
    def index(self) -> Index:
        return self._data.index

    @property
    def data(self) -> Series:
        return self._data

    @property
    def name(self) -> Hashable:
        return self._data.name

    def keys(self) -> List[str]:
        return self._data.keys().to_list()

    def lens(self) -> Series:
        return Series({
            key: len(self._data[key])
            for key in self.keys()
        })

    def __getitem__(self, item):
        """
        Return the distribution corresponding to the given key.
        """
        return self._data[item]

    def __repr__(self):

        return repr(self._data)


class DataSeriesTrendMixin(object):
    """
    Mixin for Series of ordered/numeric data distributions that support
    adjacent-pair increase probabilities and permutation trend tests.
    """
    _data: Series

    def increase_probs(self: SupportsTrendSeries) -> Series:
        """
        Return a Series of probabilities of whether each adjacent pair of
        distributions increases from one to the next.
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

    def _p_trend(
            self: SupportsTrendSeries,
            increasing: bool,
            n_iter: int = 1_000
    ) -> float:
        ref_probs = self.increase_probs()
        ref_sum = ref_probs.sum()
        ref_prod = ref_probs.prod()
        unsplit_dist = self.unsplit()
        n_x = self.lens()
        results = []
        for _ in tqdm(range(n_iter)):
            sampled = self._sample_series(unsplit_dist, n_x)
            test_probs = sampled.increase_probs()
            test_sum = test_probs.sum()
            test_prod = test_probs.prod()
            if increasing:
                is_more_extreme = (ref_sum > test_sum and ref_prod > test_prod)
            else:
                is_more_extreme = (ref_sum < test_sum and ref_prod < test_prod)
            results.append(1 if is_more_extreme else 0)
        return float(Series(results).mean())

    def p_increasing(self: SupportsTrendSeries, n_iter: int = 1_000) -> float:
        """
        Return the probability that the distributions increase as the key
        increases.
        """
        return self._p_trend(increasing=True, n_iter=n_iter)

    def p_decreasing(self: SupportsTrendSeries, n_iter: int = 1_000) -> float:
        """
        Return the probability that the distributions decrease as the key
        increases.
        """
        return self._p_trend(increasing=False, n_iter=n_iter)


