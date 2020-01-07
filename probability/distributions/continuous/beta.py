from scipy.stats import beta as beta_dist, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class Beta(RVContinuous1dMixin):

    def __init__(self, alpha: float, beta: float):

        self._alpha: float = alpha
        self._beta: float = beta
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_continuous = beta_dist(self._alpha, self._beta)

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

    def __str__(self):

        return f'Beta(α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'
