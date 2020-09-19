from typing import Callable, Any


class OperatorMixin(object):

    get_name: Callable[..., str]
    operate: Callable


class AggregatorMixin(object):

    function_name: str
    aggregate: Callable[[Any], Any]


class CalculationMixin(object):

    execute: Callable[[], Any]
