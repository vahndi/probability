from numpy import array, ndarray
from scipy.stats import rv_discrete, multinomial
from typing import Iterable, List, Tuple

from probability.distributions.mixins.rv_mixins import EntropyMixin, RVSNdMixin, PMFNdMixin
from probability.utils import k_tuples_summing_to_n


class Multinomial(
    RVSNdMixin, PMFNdMixin, EntropyMixin,
    object
):

    def __init__(self, n: int, p: Iterable[float]):

        self._n: int = n
        self._p: ndarray = array(p)
        self._num_dims = len(self._p)
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = multinomial(self._n, self._p)

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

    @property
    def p(self) -> Iterable[float]:
        return self._p

    @p.setter
    def p(self, value: Iterable[float]):
        self._p = value

    def __str__(self):

        return f'Multinomial(n={self._n}, p=[{", ".join([str(i) for i in self._p])}])'

    def __repr__(self):

        return f'Multinomial(n={self._n}, p=[{", ".join([str(i) for i in self._p])}])'
