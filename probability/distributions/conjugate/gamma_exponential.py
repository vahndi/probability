from probability.distributions.mixins.conjugate_mixin import ConjugateMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class GammaExponential(RVContinuous1dMixin, ConjugateMixin):

    def __init__(self, alpha: float, beta: float):

        self._alpha = alpha
        self._beta = beta
        self._reset_distribution()



