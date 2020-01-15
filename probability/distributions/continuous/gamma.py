from scipy.stats import rv_continuous, gamma

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class Gamma(RVContinuous1dMixin, object):

    _parametrization: str

    def __init__(self, alpha: float, beta: float, parametrization: str = 'αβ'):

        assert parametrization in ('αβ', 'kθ')
        self._alpha: float = alpha
        self._beta: float = beta
        self._parametrization: str = parametrization
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = gamma(a=self._alpha, scale=1 / self._beta)

    @staticmethod
    def from_alpha_beta(alpha: float, beta: float) -> 'Gamma':

        return Gamma(alpha=alpha, beta=beta, parametrization='αβ')

    @staticmethod
    def from_k_theta(k: float, theta: float) -> 'Gamma':

        return Gamma(alpha=k, beta=1 / theta, parametrization='kθ')

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self._reset_distribution()

    @property
    def k(self) -> float:
        return self._alpha

    @k.setter
    def k(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def theta(self):
        return 1 / self._beta

    @theta.setter
    def theta(self, value: float):
        self._beta = 1 / value
        self._reset_distribution()

    def __str__(self):

        if self._parametrization == 'αβ':
            return f'Gamma(α={self._alpha}, β={self._beta})'
        elif self._parametrization == 'kθ':
            return f'Gamma(k={self._alpha}, θ={1 / self._beta})'

    def __repr__(self):

        if self._parametrization == 'αβ':
            return f'Gamma(alpha={self._alpha}, beta={self._beta})'
        elif self._parametrization == 'kθ':
            return f'Gamma(k={self._alpha}, theta={1 / self._beta})'
