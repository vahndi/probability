from scipy.stats import beta as beta_scipy

from probability.distributions.mixins.rv_continuous_mixin import RVContinuousMixin


class Beta(RVContinuousMixin):

    def __init__(self, alpha: float, beta: float):

        self._alpha = alpha
        self._beta = beta
        self._distribution = beta_scipy(alpha, beta)

    def __str__(self):

        return f'Beta(α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'
