from probability.calculations.calculation import DistributionCalculation, \
    BinaryOperatorCalculation, CalculationInput, UnaryOperatorCalculation
from probability.calculations.operators import MultiplyDistributions, \
    ComplementDistribution


class CalculableMixin(object):

    def __mul__(self, other) -> DistributionCalculation:

        if isinstance(other, DistributionCalculation):
            name_2 = other.name
        else:
            name_2 = str(other)

        return BinaryOperatorCalculation(
            calc_input_1=CalculationInput(name=str(self), value=self),
            calc_input_2=CalculationInput(name=name_2, value=other),
            operator=MultiplyDistributions
        )

    def __rsub__(self, other):

        if (
                (isinstance(other, int) or isinstance(other, float)) and
                other == 1
        ):
            return UnaryOperatorCalculation(
                calc_input=CalculationInput(name=str(self), value=self),
                operator=ComplementDistribution
            )
        raise NotImplementedError
