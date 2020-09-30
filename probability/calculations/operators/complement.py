from pandas import Series, DataFrame

from probability.calculations.mixins import OperatorMixin


class Complement(
    OperatorMixin,
    object
):

    @classmethod
    def get_name(cls, name: str):
        return f'1 - {name}'

    @staticmethod
    def operate(value):

        if type(value) in (int, float):
            return 1 - value
        elif isinstance(value, Series):
            return (1 - value).rename(f'1 - {value.name}')
        elif isinstance(value, DataFrame):
            result = 1 - value
            result = result.rename(columns=lambda c: f'1 - {c}')
            return result
        else:
            raise TypeError(
                'value_1 must be int, float, Series or DataFrame'
            )