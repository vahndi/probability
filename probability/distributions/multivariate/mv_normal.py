from numpy import array, ndarray
from scipy.stats import multivariate_normal, rv_continuous

from probability.custom_types import FloatOrFloatArray1d, FloatOrFloatArray2d
from probability.distributions.mixins.rv_mixins import RVSNdMixin, Entropy1dMixin, PDFNdMixin, CDFContinuousNdMixin


class MVNormal(
    RVSNdMixin, CDFContinuousNdMixin, PDFNdMixin, Entropy1dMixin,
    object
):

    def __init__(self, mu: FloatOrFloatArray1d, sigma: FloatOrFloatArray2d):

        self._mu: ndarray = array(mu)
        if isinstance(mu, float):
            self._num_dims = 1
        else:
            self._num_dims = len(self._mu)
        self._sigma: ndarray = array(sigma)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = multivariate_normal(mean=self._mu, cov=self._sigma)

    @property
    def mu(self) -> ndarray:
        return self._mu

    @mu.setter
    def mu(self, value: FloatOrFloatArray1d):
        self._mu = value

    @property
    def sigma(self) -> ndarray:
        return self._sigma

    @sigma.setter
    def sigma(self, value: FloatOrFloatArray2d):
        self._sigma = value

    def __str__(self):

        return f'MVNormal(μ={self._mu}, Σ={self._sigma.tolist()})'

    def __repr__(self):

        return f'MVNormal(mu={self._mu}, sigma={self._sigma.tolist()})'
