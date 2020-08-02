from pandas import Series
from scipy.stats import dirichlet
from scipy.stats._multivariate import multi_rv_generic

from probability.custom_types import FloatArray1d
from probability.distributions.mixins.rv_mixins import RVSNdMixin, PDFNdMixin, \
    EntropyMixin, MeanNdMixin, VarNdMixin


class Dirichlet(
    RVSNdMixin, PDFNdMixin, EntropyMixin, MeanNdMixin, VarNdMixin,
    object
):

    def __init__(self, alpha: FloatArray1d):

        if not isinstance(alpha, Series):
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
