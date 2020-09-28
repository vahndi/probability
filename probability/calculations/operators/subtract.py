from probability.calculations.operators.binary_operator import BinaryOperator
from probability.custom_types.calculation_types import CalculationValue


class Subtract(
    BinaryOperator,
    object
):

    symbol = '-'

    @staticmethod
    def operate(
            value_1: CalculationValue, value_2: CalculationValue,
            value_1_calc: bool, value_2_calc: bool
    ) -> CalculationValue:
        raise NotImplementedError
