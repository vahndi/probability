from scipy.stats import laplace, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class Laplace(RVContinuous1dMixin):

    def __init__(self, mu: float, b: float):
        self._mu: float = mu
        self._b: float = b
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = laplace(self._mu, self._b)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self._reset_distribution()

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        self._b = value
        self._reset_distribution()

    def __str__(self):
        return f'Laplace(μ={self._mu}, b={self._b})'

    def __repr__(self):
        return f'Laplace(mu={self._mu}, b={self._b})'
