from typing import Tuple, Any, Callable

from pandas import Series, DataFrame

from probability.calculations.mixins import OperatorMixin
from probability.custom_types.calculation_types import CalculationValue


class BinaryOperator(OperatorMixin):

    symbol: str
    operator: Callable[[Any, Any], Any]
    pandas_op_name: str

    @classmethod
    def get_name(cls, name_1: str, name_2: str):

        return f'{name_1} {cls.symbol} {name_2}'

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

    @classmethod
    def operate(
            cls,
            value_1: CalculationValue, value_2: CalculationValue,
            value_1_calc: bool, value_2_calc: bool
    ) -> CalculationValue:

        l1, r1, l2, r2 = BinaryOperator.get_parens(value_1_calc, value_2_calc)

        if type(value_1) in (int, float):
            if type(value_2) in (int, float):
                return cls.operator(value_1, value_2)
            elif isinstance(value_2, Series):
                result = cls.operator(value_1, value_2)
                result.name = (
                    f'{l1}{value_1}{r1} '
                    f'{cls.symbol} '
                    f'{l2}{value_2.name}{r2}'
                )
                return result
            elif isinstance(value_2, DataFrame):
                result = cls.operator(value_1, value_2)
                result.columns = [
                    f'{l1}{value_1}{r1} {cls.symbol} {l2}{name_2}{r2}'
                    for name_2 in value_2.columns
                ]
                return result
            else:
                raise TypeError(
                    'value_2 must be int, float, Series or DataFrame'
                )
        elif isinstance(value_1, Series):
            if type(value_2) in (int, float):
                result = cls.operator(value_1, value_2)
                result.name = (
                    f'{l1}{value_1.name}{r1} '
                    f'{cls.symbol} '
                    f'{l2}{value_2}{r2}')
                return result
            elif isinstance(value_2, Series):
                result = cls.operator(value_1, value_2)
                result.name = (
                    f'{l1}{value_1.name}{r1} '
                    f'{cls.symbol} '
                    f'{l2}{value_2.name}{r2}')
                return result
            elif isinstance(value_2, DataFrame):
                result = getattr(
                    value_2, cls.pandas_op_name
                )(value_1, axis=0)
                result.columns = [
                    f'{l1}{value_1.name}{r1} {cls.symbol} {l2}{column}{r2}'
                    for column in value_2.columns
                ]
                return result
            else:
                raise TypeError(
                    'value_2 must be int, float, Series or DataFrame'
                )
        elif isinstance(value_1, DataFrame):
            if type(value_2) in (int, float):
                result = cls.operator(value_1, value_2)
                result.columns = [
                    f'{l1}{column}{r1} {cls.symbol} {l2}{value_2}{r2}'
                    for column in value_1.columns
                ]
                return result
            elif isinstance(value_2, Series):
                result = getattr(
                    value_1, cls.pandas_op_name
                )(value_2, axis=0)
                result.columns = [
                    f'{l1}{column}{r1} {cls.symbol} {l2}{value_2.name}{r2}'
                    for column in value_1.columns
                ]
                return result
            elif isinstance(value_2, DataFrame):
                if value_1.shape[1] != value_2.shape[1]:
                    raise ValueError(
                        'Can only operate 2 dataframes together with same '
                        'number of columns'
                    )
                result = DataFrame.from_dict({
                    f'{l1}{col_1}{r1} {cls.symbol} {l2}{col_2}{r2}':
                        cls.operator(value_1[col_1], value_2[col_2])
                    for col_1, col_2 in zip(value_1.columns, value_2.columns)
                })
                return result
            else:
                raise TypeError(
                    'value_2 must be int, float, Series or DataFrame'
                )
        else:
            raise TypeError(
                'value_1 must be int, float, Series or DataFrame'
            )
