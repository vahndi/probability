from scipy.stats import lomax, rv_continuous

from probability.distributions.mixins.attributes import AlphaFloatDMixin, \
    LambdaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin


class Lomax(
    RVContinuous1dMixin,
    AlphaFloatDMixin,
    LambdaFloatDMixin,
    CalculableMixin,
    object
):
    """
    The Lomax distribution, conditionally also called the Pareto Type II
    distribution, is a heavy-tail probability distribution used in business,
    economics, actuarial science, queueing theory and Internet traffic modeling.

    It is essentially a Pareto distribution that has been shifted so that its
    support begins at zero.

    https://en.wikipedia.org/wiki/Lomax_distribution
    """
    def __init__(self, lambda_: float, alpha: float):

        self._lambda: float = lambda_
        self._alpha: float = alpha
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_continuous = lomax(
            c=self._alpha, scale=self._lambda
        )

    def __str__(self):

        return f'Lomax(λ={self._lambda}, α={self._alpha})'

    def __repr__(self):

        return f'Lomax(lambda_={self._lambda}, alpha={self._alpha})'
