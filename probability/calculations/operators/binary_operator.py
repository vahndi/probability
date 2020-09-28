from typing import Tuple

from probability.calculations.operators.complement import OperatorMixin
from probability.custom_types.calculation_types import CalculationValue


class BinaryOperator(OperatorMixin):

    symbol: str

    @classmethod
    def get_name(cls, name_1: str, name_2: str):

        return f'{name_1} {cls.symbol} {name_2}'

    @staticmethod
    def operate(
            value_1: CalculationValue, value_2: CalculationValue,
            value_1_calc: bool, value_2_calc: bool
    ) -> CalculationValue:

        raise NotImplementedError

    @staticmethod
    def get_parens(
            value_1_calc: bool, value_2_calc: bool
    ) -> Tuple[str, str, str, str]:

        if value_1_calc:
            l1 = '('
            r1 = ')'
        else:
            l1 = ''
            r1 = ''
        if value_2_calc:
            l2 = '('
            r2 = ')'
        else:
            l2 = ''
            r2 = ''

        return l1, r1, l2, r2