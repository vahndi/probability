from math import sqrt

from scipy.stats import t, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class StudentsT(RVContinuous1dMixin):

    def __init__(self, nu: float, mu: float = 0, sigma: float = 1, sigma_sq: float = None):
        """
        :param nu: Number of degrees of freedom.
        """

        assert any([sigma, sigma_sq]) and not all([sigma, sigma_sq])

        self._nu: float = nu
        self._mu: float = mu
        if sigma is not None:
            self._sigma: float = sigma
            self._parametrization: str = 'μσ'
        else:
            self._sigma: float = sqrt(sigma_sq)
            self._parametrization: str = 'μσ²'
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = t(self._nu, loc=self._mu, scale=self._sigma)

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value: float):

        self._nu = value
        self._reset_distribution()

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self._reset_distribution()

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float):
        self._sigma = value
        self._reset_distribution()

    @property
    def sigma_sq(self) -> float:
        return self._sigma ** 2

    @sigma_sq.setter
    def sigma_sq(self, value: float):
        self._sigma = sqrt(value)
        self._reset_distribution()

    def __str__(self):
        if self._parametrization == 'μσ':
            return f'StudentsT($\\nu$={self._nu}, μ={self._mu}, σ={self._sigma})'
        elif self._parametrization == 'μσ²':
            return f'StudentsT($\\nu$={self._nu}, μ={self._mu}, σ²={self.sigma_sq})'

    def __repr__(self):

        if self._parametrization == 'μσ':
            return f'StudentsT(nu={self._nu}, mu={self._mu}, sigma={self._sigma})'
        elif self._parametrization == 'μσ²':
            return f'StudentsT(nu={self._nu}, mu={self._mu}, sigma_sq={self.sigma_sq})'
