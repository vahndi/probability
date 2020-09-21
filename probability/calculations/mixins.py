from typing import Callable, Any, List


class OperatorMixin(object):

    get_name: Callable[..., str]
    operate: Callable


class AggregatorMixin(object):

    function_name: str
    aggregate: Callable[[Any], Any]


class ProbabilityCalculationMixin(object):

    context: Any
    input_calcs: List['ProbabilityCalculationMixin']
    set_context: Callable[[Any], 'ProbabilityCalculationMixin']

