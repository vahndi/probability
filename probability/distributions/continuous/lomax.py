from typing import Optional

from scipy.stats import lomax, rv_continuous

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import AlphaFloatDMixin, \
    LambdaFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


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

    def mode(self) -> float:
        return 0.0

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float:
        return self.ppf().at(0.99)

    @staticmethod
    def fit(data: FloatIterable,
            lambda_: Optional[float] = None,
            alpha: Optional[float] = None):
        """
        Fit a Lomax distribution to the data.

        :param data: Iterable of data to fit to.
        :param lambda_: Optional fixed value for lambda_.
        :param alpha: Optional fixed value for alpha.
        """
        kwargs = {}
        for arg, kw in zip(
            (lambda_, alpha),
            ('fscale', 'fc')
        ):
            if arg is not None:
                kwargs[kw] = arg
        c, loc, scale = lomax.fit(data=data, **kwargs)
        return Lomax(lambda_=scale, alpha=c)

    def __str__(self):

        return (
            f'Lomax('
            f'λ={num_format(self._lambda, 3)}, '
            f'α={num_format(self._alpha, 3)})'
        )

    def __repr__(self):

        return f'Lomax(lambda_={self._lambda}, alpha={self._alpha})'

    def __eq__(self, other: 'Lomax') -> bool:

        return (
            abs(self._alpha - other._alpha) < 1e-10 and
            abs(self._lambda - other._lambda) < 1e-10
        )

    def __ne__(self, other: 'Lomax') -> bool:

        return not self.__eq__(other)
