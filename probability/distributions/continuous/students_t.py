from math import sqrt

from scipy.stats import t, rv_continuous

from probability.distributions.mixins.attributes import MuFloatDMixin, \
    SigmaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class StudentsT(
    RVContinuous1dMixin,
    MuFloatDMixin,
    SigmaFloatDMixin,
    CalculableMixin,
    object
):
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
    def sigma_sq(self) -> float:
        return self._sigma ** 2

    @sigma_sq.setter
    def sigma_sq(self, value: float):
        self._sigma = sqrt(value)
        self._reset_distribution()

    def __str__(self):
        if self._parameterization == 'μσ':
            return f'StudentsT(' \
                   f'$\\nu$={num_format(self._nu, 3)}, ' \
                   f'μ={num_format(self._mu, 3)}, ' \
                   f'σ={num_format(self._sigma, 3)})'
        elif self._parameterization == 'μσ²':
            return f'StudentsT(' \
                   f'$\\nu$={num_format(self._nu, 3)}, ' \
                   f'μ={num_format(self._mu, 3)}, ' \
                   f'σ²={num_format(self.sigma_sq, 3)})'

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
