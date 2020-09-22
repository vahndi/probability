from typing import overload, Union, Mapping, Any

from pandas import Series

from probability.calculations.calculations import ProbabilityCalculation, \
    BinaryOperatorCalculation, SampleCalculation, \
    ValueCalculation, UnaryOperatorCalculation
from probability.calculations.context import CalculationContext
from probability.calculations.operators import Multiply, Complement
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

    def __mul__(self, other):

        if isinstance(other, ProbabilityCalculation):
            context = other.context
            input_2 = other
        else:
            context = CalculationContext()
            if isinstance(other, float):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_2 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series):
                return Series({
                    key: (self * value).set_context(context)
                    for key, value in other.items()
                })
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin or float'
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

        return CalculableMixin.__mul__(other, self)

    def __rsub__(self, other):

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
        raise NotImplementedError
