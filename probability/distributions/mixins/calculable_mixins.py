from typing import overload, Union, Mapping, Any

from pandas import Series, DataFrame

from probability.calculations.binary_operator_calculation import \
    BinaryOperatorCalculation
from probability.calculations.context import CalculationContext
from probability.calculations.operators.complement import Complement
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.operators.divide import Divide
from probability.calculations.operators.add import Add
from probability.calculations.operators.multiply import Multiply
from probability.calculations.sample_calculation import SampleCalculation
from probability.calculations.unary_operator_calculation import \
    UnaryOperatorCalculation
from probability.calculations.value_calculation import ValueCalculation
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin


class CalculableMixin(object):

    @overload
    def __mul__(
            self, other: Union[int, float, RVS1dMixin, RVSNdMixin]
    ) -> ProbabilityCalculationMixin:

        pass

    @overload
    def __mul__(
            self, other: Series
    ) -> Union[Series, Mapping[Any, ProbabilityCalculationMixin]]:

        pass

    @overload
    def __mul__(
            self, other: DataFrame
    ) -> Union[DataFrame,
               Mapping[Any, Mapping[Any, ProbabilityCalculationMixin]]]:

        pass

    def __mul__(self, other):
        """
        Multiply the Distribution by a float, distribution, Series or DataFrame.

        :param other: The multiplier. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced. Use
                      `sync_context` if syncing is needed.
        """
        if isinstance(other, ProbabilityCalculationMixin):
            context = other.context
            input_2 = other
        else:
            context = CalculationContext()
            if type(other) in (int, float):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_2 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series):
                return Series({
                    key: self * value
                    for key, value in other.iteritems()
                })
            elif isinstance(other, DataFrame):
                return DataFrame({
                    column: {key: self * value
                             for key, value in other[column].iteritems()}
                    for column in other.columns
                })
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        input_1 = SampleCalculation(
            calc_input=self,
            context=context
        )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=input_2,
            operator=Multiply,
            context=context
        )

    def __rmul__(self, other):

        if isinstance(other, ProbabilityCalculationMixin):
            context = other.context
            input_1 = other
        else:
            context = CalculationContext()
            if type(other) in (int, float):
                input_1 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_1 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series) or isinstance(other, DataFrame):
                return other * self
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        input_2 = SampleCalculation(
            calc_input=self,
            context=context
        )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=input_2,
            operator=Multiply,
            context=context
        )

    def __rsub__(self, other) -> ProbabilityCalculationMixin:
        """
        Used for returning the synced complement of a distribution.

        :param other: Must be 1
        """
        if (
                (isinstance(other, int) or isinstance(other, float)) and
                other == 1
        ):
            context = CalculationContext()
            return UnaryOperatorCalculation(
                calc_input=SampleCalculation(calc_input=self, context=context),
                operator=Complement,
                context=context
            )
        else:
            raise NotImplementedError

    def __add__(self, other):

        if isinstance(other, ProbabilityCalculationMixin):
            context = other.context
            input_2 = other
        else:
            context = CalculationContext()
            if isinstance(other, float):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_2 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series) or isinstance(other, DataFrame):
                return other * self
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        input_1 = SampleCalculation(
            calc_input=self,
            context=context
        )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=input_2,
            operator=Add,
            context=context
        )

    def __radd__(self, other):

        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other):
        """
        Multiply the Distribution by a float, distribution, Series or DataFrame.

        :param other: The divisor. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced. Use
                      `sync_context` if syncing is needed.
        """
        if isinstance(other, ProbabilityCalculationMixin):
            context = other.context
            input_2 = other
        else:
            context = CalculationContext()
            if type(other) in (int, float):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_2 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series):
                return Series({
                    key: self / value
                    for key, value in other.iteritems()
                })
            elif isinstance(other, DataFrame):
                return DataFrame({
                    column: {key: self / value
                             for key, value in other[column].iteritems()}
                    for column in other.columns
                })
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        input_1 = SampleCalculation(
            calc_input=self,
            context=context
        )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=input_2,
            operator=Divide,
            context=context
        )

    def __rtruediv__(self, other):

        if isinstance(other, ProbabilityCalculationMixin):
            context = other.context
            input_1 = other
        else:
            context = CalculationContext()
            if type(other) in (int, float):
                input_1 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_1 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series) or isinstance(other, DataFrame):
                return other / self
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        input_2 = SampleCalculation(
            calc_input=self,
            context=context
        )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=input_2,
            operator=Divide,
            context=context
        )
