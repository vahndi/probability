from probability.distributions import Beta
from probability.distributions.mixins.prior_mixin import PriorMixin


class BetaPrior(PriorMixin, Beta):

    def __init__(self, alpha: float, beta: float):

        Beta.__init__(self, alpha=alpha, beta=beta)
        self._x_label: str = 'x'
        self._y_label: str = 'p(x|α,β)'
