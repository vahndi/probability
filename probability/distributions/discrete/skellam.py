from typing import Union

from scipy.stats import rv_discrete, skellam

from probability.custom_types.external_custom_types import FloatArray1d
from probability.distributions.mixins.attributes import NIntDMixin, PFloatDMixin
from probability.distributions.mixins.calculable_mixin import CalculableMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin
from probability.utils import num_format, is_scalar


class Skellam(
    RVDiscrete1dMixin,
    NIntDMixin,
    PFloatDMixin,
    CalculableMixin,
    object
):
    """
    The Skellam distribution is the discrete probability distribution of the
    difference N1 - N2 of two statistically independent random variables N1 and
    N2, each Poisson-distributed with respective expected values μ1 and μ2.
    It is useful in describing the statistics of the difference of two images
    with simple photon noise, as well as describing the point spread
    distribution in sports where all scored points are equal, such as baseball,
    hockey and soccer.

    The distribution is also applicable to a special case of the difference of
    dependent Poisson random variables, but just the obvious case where the two
    variables have a common additive random contribution which is cancelled by
    the differencing: see Karlis & Ntzoufras (2003) for details and an
    application.

    https://en.wikipedia.org/wiki/Skellam_distribution
    """
    def __init__(self, mu_1: float, mu_2: float):
        """
        Create a new Skellam distribution.

        :param mu_1: Mean of the distribution to subtract from.
        :param mu_2: Mean of the distribution to subtract.
        """
        self._mu_1: float = mu_1
        self._mu_2: float = mu_2
        self._reset_distribution()

    def _reset_distribution(self):

        self._distribution: rv_discrete = skellam(self._mu_1, self._mu_2)

    @property
    def lower_bound(self) -> int:
        return int(round(self.ppf().at(0.005)))

    @property
    def upper_bound(self) -> int:
        return int(round(self.ppf().at(0.995)))

    @staticmethod
    def fit(data: FloatArray1d, **kwargs) -> 'RVDiscrete1dMixin':
        raise NotImplementedError

    @staticmethod
    def fits(data: FloatArray1d, **kwargs) -> 'RVDiscrete1dMixin':
        raise NotImplementedError

    def __str__(self):

        return (
            f'Skellam('
            f'μ1={num_format(self._mu_1, 3)}, '
            f'μ1={num_format(self._mu_2, 3)})'
        )

    def __repr__(self):

        return f'Skellam(mu_1={self._mu_1}, mu_2={self._mu_2})'

    def __eq__(
            self, other: Union['Skellam', int, float]
    ) -> Union[bool, float]:

        if is_scalar(other):
            return self.pmf().at(other)
        else:
            return (
                self._mu_1 == other._mu_1 and
                self._mu_2 == other._mu_2
            )

    def __ne__(
            self, other: Union['Skellam', int, float]
    ) -> Union[bool, float]:

        if is_scalar(other):
            return 1 - self.pmf().at(other)
        else:
            return not self.__eq__(other)
