from typing import List, Tuple, Union

from pandas import Series
from scipy.stats import rv_discrete, multinomial

from probability.custom_types import FloatArray1d
from probability.distributions.discrete import Binomial
from probability.distributions.mixins.rv_mixins import EntropyMixin, \
    RVSNdMixin, PMFNdMixin
from probability.utils import k_tuples_summing_to_n


class Multinomial(
    RVSNdMixin, PMFNdMixin, EntropyMixin,
    object
):
    """
    https://en.wikipedia.org/wiki/Multinomial_distribution
    """
    def __init__(self, n: int, p: Union[FloatArray1d, dict]):
        """
        Create a new Multinomial distribution.

        :param n: Number of trials.
        :param p: Probability of each outcome in any given trial.
        """
        self._n: int = n
        if isinstance(p, dict):
            p = Series(p)
        elif not isinstance(p, Series):
            p = Series(
                data=p,
                index=[f'x{k}' for k in range(1, len(p) + 1)]
            )
        self._p: Series = p
        self._num_dims = len(self._p)
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = multinomial(
            self._n, self._p.values
        )

    def permutations(self) -> List[Tuple[int]]:
        """
        Return all possible permutations of X.
        """
        return k_tuples_summing_to_n(k=len(self._p), n=self._n)

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value
        self._reset_distribution()

    @property
    def p(self) -> Series:
        return self._p

    @p.setter
    def p(self, value: FloatArray1d):

        if not isinstance(value, Series):
            value = Series(
                data=value,
                index=self._p.index
            )
        self._p = value
        self._reset_distribution()

    def __str__(self):

        p = ', '.join([f'{k}={v}' for k, v in self._p.items()])
        return f'Multinomial({p})'

    def __repr__(self):

        p = ', '.join([f'{k}={v}' for k, v in self._p.items()])
        return f'Multinomial({p})'

    def __getitem__(self, item) -> Binomial:

        return Binomial(
            n=self._n, p=self._p[item]
        )
