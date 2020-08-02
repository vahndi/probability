from typing import Union

from pandas import Series
from scipy.stats import dirichlet
from scipy.stats._multivariate import multi_rv_generic

from probability.custom_types import FloatArray1d
from probability.distributions.continuous import Beta
from probability.distributions.mixins.rv_mixins import RVSNdMixin, PDFNdMixin, \
    EntropyMixin, MeanNdMixin, VarNdMixin


class Dirichlet(
    RVSNdMixin, PDFNdMixin, EntropyMixin, MeanNdMixin, VarNdMixin,
    object
):
    """
    https://en.wikipedia.org/wiki/Dirichlet_distribution
    """
    def __init__(self, alpha: Union[FloatArray1d, dict]):

        if isinstance(alpha, dict):
            alpha = Series(alpha)
        elif not isinstance(alpha, Series):
            alpha = Series(
                data=alpha,
                index=[f'α{k}' for k in range(1, len(alpha) + 1)]
            )
        self._alpha: Series = alpha
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
