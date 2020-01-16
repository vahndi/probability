from scipy.stats import t, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class StudentsT(RVContinuous1dMixin):

    def __init__(self, nu: float):
        """
        :param nu: Number of degrees of freedom.
        """
        self._nu = nu
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = t(self._nu)

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value: float):

        self._nu = value
        self._reset_distribution()

    def __str__(self):
        return f'StudentsT($\\nu$={self._nu})'

    def __repr__(self):
        return f'StudentsT(nu={self._nu})'
