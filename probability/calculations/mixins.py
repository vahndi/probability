from typing import Callable, Any, List, Optional

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
