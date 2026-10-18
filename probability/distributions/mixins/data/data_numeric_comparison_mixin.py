from typing import TypeVar, Union, TYPE_CHECKING

from pandas import Series

from probability.distributions.mixins.calculable_mixin import CalculableMixin

if TYPE_CHECKING:
    from probability.distributions.data.boolean import Boolean

DNCM = TypeVar('DNCM', bound='DataNumericComparisonMixin')


class DataNumericComparisonMixin(CalculableMixin, object):

    _data: Series

    def quantile(self, q: float) -> float:
        """
        Return the value at the given quantile `q` in [0, 1].
        """
        return float(self._data.quantile(q))

    def where_eq(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data == value])

    def where_ne(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data != value])

    def where_gt(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data > value])

    def where_lt(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data < value])

    def where_ge(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data >= value])

    def where_le(self: DNCM, value: Union[int, float]) -> DNCM:

        return type(self)(data=self._data.loc[self._data <= value])

    def between(
            self: DNCM,
            low: Union[int, float],
            high: Union[int, float],
            inclusive: str = 'both'
    ) -> DNCM:
        """
        Return a filtered distribution with values between `low` and `high`.
        """
        mask = self._data.between(low, high, inclusive=inclusive)
        return type(self)(data=self._data.loc[mask])

    def between_quantiles(
            self: DNCM,
            low_q: float = 0.0,
            high_q: float = 1.0,
            inclusive: str = 'both'
    ) -> DNCM:
        """
        Return a filtered distribution with values between the `low_q` and
        `high_q` quantiles.
        """
        low_val = self.quantile(low_q) if low_q > 0.0 else self._data.min()
        high_val = self.quantile(high_q) if high_q < 1.0 else self._data.max()
        return self.between(low_val, high_val, inclusive=inclusive)

    def _to_boolean(self, mask: Series, op_str: str, value) -> 'Boolean':
        from probability.distributions.data.boolean import Boolean
        name = f'{self._data.name}{op_str}{value}' if self._data.name else None
        return Boolean(data=mask.rename(name))

    def eq(self, value: Union[int, float]) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self == value`.
        """
        return self._to_boolean(self._data == value, '==', value)

    def ne(self, value: Union[int, float]) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self != value`.
        """
        return self._to_boolean(self._data != value, '!=', value)

    def gt(self, value: Union[int, float]) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self > value`.
        """
        return self._to_boolean(self._data > value, '>', value)

    def lt(self, value: Union[int, float]) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self < value`.
        """
        return self._to_boolean(self._data < value, '<', value)

    def ge(self, value: Union[int, float]) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self >= value`.
        """
        return self._to_boolean(self._data >= value, '>=', value)

    def le(self, value: Union[int, float]) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self <= value`.
        """
        return self._to_boolean(self._data <= value, '<=', value)

    def gt_quantile(self, q: float) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self > self.quantile(q)`.
        """
        return self._to_boolean(self._data > self.quantile(q), '>q', q)

    def ge_quantile(self, q: float) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self >= self.quantile(q)`.
        """
        return self._to_boolean(self._data >= self.quantile(q), '>=q', q)

    def lt_quantile(self, q: float) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self < self.quantile(q)`.
        """
        return self._to_boolean(self._data < self.quantile(q), '<q', q)

    def le_quantile(self, q: float) -> 'Boolean':
        """
        Return a Boolean indicator distribution for `self <= self.quantile(q)`.
        """
        return self._to_boolean(self._data <= self.quantile(q), '<=q', q)

