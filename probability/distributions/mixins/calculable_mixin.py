from operator import add, mul, sub, truediv
from typing import Any, Callable, Mapping, Type, Union, overload

from pandas import DataFrame, Series

from probability.calculations.calculation_context import CalculationContext
from probability.calculations.mixins import (
    ProbabilityCalculationMixin, SupportsCalculation, SupportsOperator
)
from probability.calculations.operators import (
    AddOperator, ComplementOperator, DivideOperator, MultiplyOperator,
    SubtractOperator
)
from probability.distributions.protocols import SupportsRVS, is_rvs
from probability.utils import is_scalar

CalculableOperand = Union[
    int, float, SupportsRVS, SupportsCalculation
]



class CalculableMixin(object):
    """
    Base class for distributions to enable calculations.
    """

    def _forward_binary_op(
            self,
            other: Any,
            builtin_op: Callable[[Any, Any], Any],
            operator_cls: Type[SupportsOperator]
    ):
        from probability.calculations.calculation_types import (
            BinaryOperatorCalculation, SampleCalculation
        )
        from probability.calculations.calculation_types.value_calculation import (
            ValueCalculation
        )

        if isinstance(other, (ProbabilityCalculationMixin, SupportsCalculation)):
            context = other.context
            input_2 = other
        else:
            context = CalculationContext()
            if is_scalar(other):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif is_rvs(other):
                input_2 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series):
                return Series({
                    key: builtin_op(self, value)
                    for key, value in other.items()
                })
            elif isinstance(other, DataFrame):
                return DataFrame({
                    column: {key: builtin_op(self, value)
                             for key, value in other[column].items()}
                    for column in other.columns
                })
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin, int, float, '
                    'Series or DataFrame'
                )

        input_1 = SampleCalculation(
            calc_input=self,
            context=context
        )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=input_2,
            operator=operator_cls,
            context=context
        )

    def _reverse_binary_op(
            self,
            other: Any,
            builtin_op: Callable[[Any, Any], Any],
            operator_cls: Type[SupportsOperator]
    ):
        from probability.calculations.calculation_types import (
            BinaryOperatorCalculation, SampleCalculation
        )
        from probability.calculations.calculation_types.value_calculation import (
            ValueCalculation
        )

        if isinstance(other, (ProbabilityCalculationMixin, SupportsCalculation)):
            context = other.context
            input_1 = other
        else:
            context = CalculationContext()
            if is_scalar(other):
                input_1 = ValueCalculation(calc_input=other, context=context)
            elif is_rvs(other):
                input_1 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, (Series, DataFrame)):
                return builtin_op(other, self)
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin, int, float, '
                    'Series or DataFrame'
                )

        input_2 = SampleCalculation(
            calc_input=self,
            context=context
        )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=input_2,
            operator=operator_cls,
            context=context
        )

    @overload
    def __mul__(
            self, other: CalculableOperand
    ) -> ProbabilityCalculationMixin: ...

    @overload
    def __mul__(
            self, other: Series
    ) -> Union[Series, Mapping[Any, ProbabilityCalculationMixin]]: ...

    @overload
    def __mul__(
            self, other: DataFrame
    ) -> Union[DataFrame,
               Mapping[Any, Mapping[Any, ProbabilityCalculationMixin]]]: ...

    def __mul__(self, other):
        """
        Multiply the Distribution by a float, distribution, Series or DataFrame.

        :param other: The multiplier. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced. Use
                      `sync_context` if syncing is needed.
        """
        return self._forward_binary_op(other, mul, MultiplyOperator)

    def __rmul__(self, other):

        return self._reverse_binary_op(other, mul, MultiplyOperator)

    def __sub__(self, other):

        return self._forward_binary_op(other, sub, SubtractOperator)

    def __rsub__(self, other) -> ProbabilityCalculationMixin:
        """
        Used for returning the synced complement of a distribution when other == 1,
        or subtracting the distribution from another operand.
        """
        from probability.calculations.calculation_types import SampleCalculation
        from probability.calculations.calculation_types.unary_operator_calculation \
            import UnaryOperatorCalculation

        if isinstance(other, (int, float)) and other == 1:
            context = CalculationContext()
            return UnaryOperatorCalculation(
                calc_input=SampleCalculation(calc_input=self, context=context),
                operator=ComplementOperator,
                context=context
            )
        return self._reverse_binary_op(other, sub, SubtractOperator)

    def __add__(self, other):

        return self._forward_binary_op(other, add, AddOperator)

    def __radd__(self, other):

        if isinstance(other, (int, float)) and other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other):
        """
        Divide the Distribution by a float, distribution, Series or DataFrame.

        :param other: The divisor. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced. Use
                      `sync_context` if syncing is needed.
        """
        return self._forward_binary_op(other, truediv, DivideOperator)

    def __rtruediv__(self, other):

        return self._reverse_binary_op(other, truediv, DivideOperator)
