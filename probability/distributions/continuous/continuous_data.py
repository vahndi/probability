from pandas import Series

from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.mixins.rv_series import RVContinuousSeries


class ContinuousData(
    RVContinuous1dMixin,
    CalculableMixin,
    object
):

    def __init__(self, data: Series):

        self._distribution: RVContinuousSeries = RVContinuousSeries(data)

    @property
    def data(self) -> Series:
        """
        Return the underlying data used to construct the Distribution.
        """
        return self._distribution.data

    @property
    def lower_bound(self) -> float:
        return self._distribution.min()

    @property
    def upper_bound(self) -> float:
        return self._distribution.max()

    @staticmethod
    def fit(data: Series) -> 'ContinuousData':
        return ContinuousData(data)

    def __repr__(self):

        return f'ContinuousData({self.data.name})'
