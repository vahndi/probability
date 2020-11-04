from typing import Union, Iterable, Optional, List

from matplotlib.axes import Axes
from mpl_format.axes.axis_utils import new_axes
from pandas import Series
from scipy.stats import dirichlet
from scipy.stats._multivariate import multi_rv_generic

from probability.custom_types.external_custom_types import FloatArray1d
from probability.distributions.continuous import Beta
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.dimension_mixins import NdMixin
from probability.distributions.mixins.plottable_mixin import \
    ContinuousPlottableNdMixin
from probability.distributions.mixins.rv_mixins import RVSNdMixin, PDFNdMixin, \
    EntropyMixin, MeanNdMixin, VarNdMixin


class Dirichlet(
    NdMixin,
    RVSNdMixin,
    PDFNdMixin,
    EntropyMixin,
    MeanNdMixin,
    VarNdMixin,
    ContinuousPlottableNdMixin,
    CalculableMixin,
    object
):
    """
    https://en.wikipedia.org/wiki/Dirichlet_distribution
    """
    def __init__(self, alpha: Union[FloatArray1d, dict, Series]):
        """
        Create a new Dirichlet distribution.

        :param alpha: List of probabilities summing to 1, or a dict mapping
                      names to probabilities. If a list is passed, names will
                      be α1, ..., αK
        """
        if isinstance(alpha, dict):
            alpha = Series(alpha)
        elif not isinstance(alpha, Series):
            names = [f'α{k}' for k in range(1, len(alpha) + 1)]
            alpha = Series(
                data=alpha,
                index=names
            )
        self._alpha: Series = alpha
        self._set_names(list(alpha.keys()))
        self._num_dims = len(alpha)
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: multi_rv_generic = dirichlet(
            alpha=self._alpha.values
        )

    @property
    def alpha(self) -> Series:

        return self._alpha

    @alpha.setter
    def alpha(self, value: FloatArray1d):

        if not isinstance(value, Series):
            value = Series(
                data=value,
                index=self._alpha.index
            )
        self._alpha = value
        self._reset_distribution()

    def __str__(self):

        params = ', '.join([f'{k}={v}' for k, v in self._alpha.items()])
        return f'Dirichlet({params})'

    def __repr__(self):

        params = ', '.join([f'{k}={v}' for k, v in self._alpha.items()])
        return f'Dirichlet({params})'

    def __getitem__(self, item) -> Beta:

        return Beta(
            alpha=self._alpha[item],
            beta=self._alpha.sum() - self._alpha[item]
        )

    def __eq__(self, other: 'Dirichlet'):

        return (
            set(self._alpha.keys()) == set(other._alpha.keys()) and
            all(
                abs(self._alpha[k] - other._alpha[k]) < 1e-10
                for k in self._alpha.keys()
            )
        )

    def plot(
            self,
            x: Iterable,
            kind: str = 'line',
            colors: Optional[List[str]] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot the function.

        :param x: Range of values of x to plot p(x) over.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param colors: Optional list of colors for each series.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        return self.pdf().plot(
            x=x, kind=kind, colors=colors, ax=ax, **kwargs
        )

