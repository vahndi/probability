from math import sqrt

from scipy.stats import t, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class StudentsT(RVContinuous1dMixin):
    """
    Student's t-distribution (or simply the t-distribution) is any member of a
    family of continuous probability distributions that arises when estimating
    the mean of a normally distributed population in situations where the sample
    size is small and the population standard deviation is unknown.

    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    def __init__(self, nu: float, mu: float = 0,
                 sigma: float = 1, sigma_sq: float = None):
        """
        Create a new Student's t distribution.

        :param nu: Number of degrees of freedom.
        """
        assert any([sigma, sigma_sq]) and not all([sigma, sigma_sq])

        self._nu: float = nu
        self._mu: float = mu
        if sigma is not None:
            self._sigma: float = sigma
            self._parameterization: str = 'μσ'
        else:
            self._sigma: float = sqrt(sigma_sq)
            self._parameterization: str = 'μσ²'
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = t(
            self._nu, loc=self._mu, scale=self._sigma
        )

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
        if self._parameterization == 'μσ':
            return f'StudentsT(' \
                   f'$\\nu$={self._nu}, ' \
                   f'μ={self._mu}, ' \
                   f'σ={self._sigma})'
        elif self._parameterization == 'μσ²':
            return f'StudentsT(' \
                   f'$\\nu$={self._nu}, ' \
                   f'μ={self._mu}, ' \
                   f'σ²={self.sigma_sq})'

    def __repr__(self):

        if self._parameterization == 'μσ':
            return f'StudentsT(' \
                   f'nu={self._nu}, ' \
                   f'mu={self._mu}, ' \
                   f'sigma={self._sigma})'
        elif self._parameterization == 'μσ²':
            return f'StudentsT(' \
                   f'nu={self._nu}, ' \
                   f'mu={self._mu}, ' \
                   f'sigma_sq={self.sigma_sq})'
