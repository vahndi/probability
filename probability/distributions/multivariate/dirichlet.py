from typing import Iterable

from numpy import array, ndarray
from scipy.stats import dirichlet
from scipy.stats._multivariate import multi_rv_generic

from probability.custom_types import FloatArray1d
from probability.distributions.mixins.rv_mixins import RVSNdMixin, PDFNdMixin, EntropyMixin, MeanNdMixin, VarNdMixin


class Dirichlet(
    RVSNdMixin, PDFNdMixin, EntropyMixin, MeanNdMixin, VarNdMixin,
    object
):

    def __init__(self, alpha: FloatArray1d):

        self._alpha: ndarray = array(alpha)
        self._num_dims = len(alpha)
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: multi_rv_generic = dirichlet(alpha=self._alpha)

    @property
    def alpha(self) -> ndarray:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Iterable[float]):

        self._alpha: ndarray = array(value)
        self._reset_distribution()
