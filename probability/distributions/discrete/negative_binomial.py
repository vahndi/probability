from scipy.stats import rv_discrete, nbinom

from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin

# TODO: reparameterize to match the wikipedia parametrization
# https://stackoverflow.com/questions/40846992/alternative-parametrization-of-the-negative-binomial-in-scipy/40855065#40855065


class NegativeBinomial(RVDiscrete1dMixin):

    def __init__(self, r: int, p: float):

        raise NotImplementedError
        self._r: int = r
        self._p: float = p
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = nbinom(self._r, self._p)

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, value: float):
        self._r = value
        self._reset_distribution()

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float):
        self._p = value
        self._reset_distribution()

    def __str__(self):

        return f'NegativeBinomial(r={self._r}, p={self._p})'

    def __repr__(self):

        return f'NegativeBinomial(r={self._r}, p={self._p})'
