from typing import Callable, Any, List, Optional

from pandas import Series, DataFrame

from probability.custom_types.calculation_types import CalculationValue


class OperatorMixin(object):

    get_name: Callable[..., str]
    operate: Callable


class AggregatorMixin(object):

    function_name: str
    aggregate: Callable[[Any], Any]


class ProbabilityCalculationMixin(object):

    name: str
    context: Any
    input_calcs: List['ProbabilityCalculationMixin']
    set_context: Callable[[Any], 'ProbabilityCalculationMixin']
    output: Callable[[Optional[int]], CalculationValue]


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