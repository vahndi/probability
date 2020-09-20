from probability.calculations.calculation import ProbabilityCalculation, \
    BinaryOperatorCalculation, SampleCalculation, \
    CalculationContext, ValueCalculation, UnaryOperatorCalculation
from probability.calculations.operators import Multiply, Complement
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin


class CalculableMixin(object):

    def __mul__(self, other) -> ProbabilityCalculation:

        if isinstance(other, ProbabilityCalculation):
            context = other.context
            input_2 = other
        else:
            context = CalculationContext()
            if isinstance(other, float):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_2 = SampleCalculation(calc_input=other, context=context)
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
