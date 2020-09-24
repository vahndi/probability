from typing import overload, Union, Mapping, Any

from pandas import Series, DataFrame

from probability.calculations.calculations import ProbabilityCalculation, \
    BinaryOperatorCalculation, SampleCalculation, \
    ValueCalculation, UnaryOperatorCalculation
from probability.calculations.context import CalculationContext
from probability.calculations.operators import Multiply, Complement, Add
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin


class CalculableMixin(object):

    @overload
    def __mul__(
            self, other: Union[float, RVS1dMixin, RVSNdMixin]
    ) -> ProbabilityCalculation:

        pass

    @overload
    def __mul__(
            self, other: Series
    ) -> Union[Series, Mapping[Any, ProbabilityCalculation]]:

        pass

    @overload
    def __mul__(
            self, other: DataFrame
    ) -> Union[DataFrame, Mapping[Any, Mapping[Any, ProbabilityCalculation]]]:

        pass

    def __mul__(self, other):
        """
        Multiply the Distribution by a float, distribution, Series or DataFrame.

        :param other: The multiplier. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced. Use
                      `sync_context` if syncing is needed.
        """
        if isinstance(other, ProbabilityCalculation):
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
            operator=Multiply,
            context=context
        )

    def __rmul__(self, other):

        if isinstance(other, ProbabilityCalculation):
            context = other.context
            input_1 = other
        else:
            context = CalculationContext()
            if isinstance(other, float):
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

    def __rsub__(self, other) -> ProbabilityCalculation:
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

        if isinstance(other, ProbabilityCalculation):
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