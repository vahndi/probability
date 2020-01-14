from scipy.stats import lomax

from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class GammaExponential(RVContinuous1dMixin, ConjugateMixin):

    def __init__(self, alpha: float, beta: float, n: int):
        """
        :param alpha:
        :param beta:
        """
        self._alpha: float = alpha
        self._beta: float = beta
        self._n: int = n

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value: float):
        self._n = value

    # def prior(self) -> Exponential:

    def __str__(self):

        return f'GammaExponential(n={self._n}, α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'GammaExponential(n={self._n}, alpha={self._alpha}, beta={self._beta})'
