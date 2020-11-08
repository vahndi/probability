from typing import List, Tuple, Union, Iterable, Optional

from matplotlib.axes import Axes
from numpy.core.records import ndarray
from pandas import Series
from scipy.stats import rv_discrete, multinomial

from probability.custom_types.external_custom_types import FloatArray1d
from probability.distributions.discrete import Binomial
from probability.distributions.functions.discrete_function_nd import \
    DiscreteFunctionNd
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.dimension_mixins import NdMixin
from probability.distributions.mixins.rv_mixins import EntropyMixin, \
    RVSNdMixin, PMFNdMixin
from probability.utils import k_tuples_summing_to_n, num_format


class Multinomial(
    NdMixin,
    RVSNdMixin,
    PMFNdMixin,
    EntropyMixin,
    DiscreteFunctionNd,
    CalculableMixin,
    object
):
    """
    https://en.wikipedia.org/wiki/Multinomial_distribution
    """
    def __init__(self, n: int, p: Union[FloatArray1d, dict, Series]):
        """
        Create a new Multinomial distribution.

        :param n: Number of trials.
        :param p: Probability of each outcome in any given trial.
        """
        self._n: int = n
        if isinstance(p, dict):
            p = Series(p)
        elif not isinstance(p, Series):
            names = [f'p{k}' for k in range(1, len(p) + 1)]
            p = Series(
                data=p,
                index=names
            )
        self._p: Series = p
        self._set_names(list(p.keys()))
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

    def plot(self, k: Union[Iterable[Iterable], ndarray],
             color: str = 'C0', ax: Optional[Axes] = None,
             **kwargs) -> Axes:
        """
        Plot the function.

        :param k: Range of values of x to plot p(x) over.
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for bar plot.
        """
        self.pmf().plot(k=k, color=color, ax=ax, **kwargs)

    def __str__(self):

        p = ', '.join([f'{k}={num_format(v, 3)}'
                       for k, v in self._p.items()])
        return f'Multinomial({p})'

    def __repr__(self):

        p = ', '.join([f'{k}={v}' for k, v in self._p.items()])
        return f'Multinomial({p})'

    def __getitem__(self, item) -> Binomial:

        return Binomial(
            n=self._n, p=self._p[item]
        )
