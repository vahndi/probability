from scipy.stats import poisson, rv_discrete

from probability.distributions.mixins.rv_discrete_mixin import RVDiscreteMixin


class Poisson(RVDiscreteMixin):

    def __init__(self, lambda_: int):

        self._lambda: int = lambda_
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = poisson(self._lambda)

    @property
    def lambda_(self) -> int:
        return self._p

    @lambda_.setter
    def lambda_(self, value: int):
        self._lambda = value
        self._reset_distribution()

    def __str__(self):

        return f'Poisson(λ={self._lambda})'

    def __repr__(self):

        return f'Poisson(lambda_={self._lambda})'
