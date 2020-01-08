from scipy.stats import poisson, rv_discrete

from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


class Poisson(RVDiscrete1dMixin):

    def __init__(self, lambda_: int):

        self._lambda: int = lambda_
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = poisson(self._lambda)

    @property
    def lambda_(self) -> int:
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: int):
        self._lambda = value
        self._reset_distribution()

    def __str__(self):

        return f'Poisson(λ={self._lambda})'

    def __repr__(self):

        return f'Poisson(lambda_={self._lambda})'
