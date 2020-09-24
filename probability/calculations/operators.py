from pandas import Series, DataFrame

from probability.calculations.mixins import OperatorMixin
from probability.custom_types.calculation_types import CalculationValue


class BinaryOperator(OperatorMixin):

    symbol: str

    @classmethod
    def get_name(cls, name_1: str, name_2: str):

        return f'{name_1} {cls.symbol} {name_2}'


class Add(
    BinaryOperator,
    object
):

    symbol = '+'

    @staticmethod
    def operate(value_1, value_2,
                num_samples: int = None):
        raise NotImplementedError


class Subtract(
    BinaryOperator,
    object
):

    symbol = '-'

    @staticmethod
    def operate(value_1, value_2,
                num_samples: int = None):
        raise NotImplementedError


class Multiply(
    BinaryOperator,
    object
):

    symbol = '*'

    @staticmethod
    def operate(
            value_1: CalculationValue, value_2: CalculationValue
    ) -> CalculationValue:

        if type(value_1) in (int, float):
            if type(value_2) in (int, float):
                return value_1 * value_2
            elif isinstance(value_2, Series):
                result = value_1 * value_2
                result.name = f'{value_1} * {value_2.name}'
                return result
            elif isinstance(value_2, DataFrame):
                result = value_1 * value_2
                result.columns = [f'{value_1} * {name_2}'
                                  for name_2 in value_2.columns]
                return result
            else:
                raise TypeError(
                    'value_2 must be int, float, Series or DataFrame'
                )
        elif isinstance(value_1, Series):
            if type(value_2) in (int, float):
                result = value_1 * value_2
                result.name = f'{value_1.name} * {value_2}'
                return result
            elif isinstance(value_2, Series):
                result = value_1 * value_2
                result.name = f'{value_1.name} * {value_2.name}'
                return result
            elif isinstance(value_2, DataFrame):
                result = value_2.mul(value_1, axis=0)
                result.columns = [f'{value_1.name} * {column}'
                                  for column in value_2.columns]
                return result
            else:
                raise TypeError(
                    'value_2 must be int, float, Series or DataFrame'
                )
        elif isinstance(value_1, DataFrame):
            if type(value_2) in (int, float):
                result = value_1 * value_2
                result.columns = [f'{column} * {value_2}'
                                  for column in value_1.columns]
                return result
            elif isinstance(value_2, Series):
                result = value_1.mul(value_2, axis=0)
                result.columns = [f'{column} * {value_2.name}'
                                  for column in value_1.columns]
                return result
            elif isinstance(value_2, DataFrame):
                result = DataFrame.from_dict({
                    f'{col_1} * {col_2}': value_1[col_1] * value_2[col_2]
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


class Divide(
    BinaryOperator,
    object
):

    symbol = '/'

    @staticmethod
    def operate(value_1, value_2):
        raise NotImplementedError


class Complement(
    OperatorMixin,
    object
):

    @classmethod
    def get_name(cls, name: str):
        return f'1 - {name}'

    @staticmethod
    def operate(value):

        return 1 - value
