from scipy.stats import invgamma, rv_continuous

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin


class InvGamma(RVContinuous1dMixin, object):
    """
    The inverse gamma distribution is a two-parameter family of continuous
    probability distributions on the positive real line, which is the
    distribution of the reciprocal of a variable distributed according to the
    gamma distribution.

    Perhaps the chief use of the inverse gamma distribution is in Bayesian
    statistics, where the distribution arises as the marginal posterior
    distribution for the unknown variance of a normal distribution, if an
    uninformative prior is used, and as an analytically tractable conjugate
    prior, if an informative prior is required.

    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    def __init__(self, alpha: float, beta: float):

        self._alpha: float = alpha
        self._beta: float = beta
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = invgamma(a=self._alpha, scale=self._beta)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._reset_distribution()

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self._reset_distribution()

    def __str__(self):

        return f'InvGamma(α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'InvGamma(alpha={self._alpha}, beta={self._beta})'
