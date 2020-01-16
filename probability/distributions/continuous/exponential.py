from scipy.stats import expon, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class Exponential(RVContinuous1dMixin):

    def __init__(self, lambda_: float):

        self._lambda: float = lambda_
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = expon(loc=0, scale=1 / self._lambda)

    @property
    def lambda_(self) -> float:
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: float):
        self._lambda = value

    def __str__(self):

        return f'Exponential(λ={self._lambda})'

    def __repr__(self):

        return f'Exponential(lambda_={self._lambda})'
