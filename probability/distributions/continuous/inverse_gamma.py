from typing import Optional

from scipy.stats import invgamma, rv_continuous

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import AlphaFloatDMixin, \
    BetaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class InverseGamma(
    RVContinuous1dMixin,
    AlphaFloatDMixin,
    BetaFloatDMixin,
    CalculableMixin,
    object
):
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
        self._distribution: rv_continuous = invgamma(
            a=self._alpha, scale=self._beta
        )

    def mode(self) -> float:
        return self.beta / (self.alpha + 1)

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.isf().at(0.01)

    @staticmethod
    def fit(data: FloatIterable,
            alpha: Optional[float] = None,
            beta: Optional[float] = None) -> 'InverseGamma':
        """
        Fit an Inverse Gamma distribution to the data.

        :param data: Iterable of data to fit to.
        :param alpha: Optional fixed value for alpha.
        :param beta: Optional fixed value for beta.
        """
        kwargs = {}
        for arg, kw in zip(
            (alpha, beta),
            ('fa', 'fb')
        ):
            if arg is not None:
                kwargs[kw] = arg
        alpha, loc, scale = invgamma.fit(data=data, floc=0, **kwargs)
        return InverseGamma(alpha=alpha, beta=scale)

    def __str__(self):

        return (
            f'InverseGamma('
            f'α={num_format(self._alpha, 3)}, '
            f'β={num_format(self._beta, 3)})'
        )

    def __repr__(self):

        return f'InverseGamma(alpha={self._alpha}, beta={self._beta})'

    def __eq__(self, other: 'InverseGamma') -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._beta - other._beta) < 1e-10
        )

    def __ne__(self, other: 'InverseGamma') -> bool:

        return not self.__eq__(other)
