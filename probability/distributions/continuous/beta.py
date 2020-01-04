from scipy.stats import beta as beta_dist, rv_continuous

from probability.distributions.mixins.rv_continuous_mixin import RVContinuousMixin


class Beta(RVContinuousMixin):

    def __init__(self, alpha: float, beta: float):

        self._alpha: float = alpha
        self._beta: float = beta
        self._distribution: rv_continuous = beta_dist(alpha, beta)

    def __str__(self):

        return f'Beta(α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'
