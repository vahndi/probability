from typing import Callable, Any, List, Optional

from probability.custom_types.calculation_types import CalculationValue


class OperatorMixin(object):

    get_name: Callable[..., str]
    operate: Callable


class ProbabilityCalculationMixin(object):

    name: str

    @property
    def input_calcs(self) -> List['ProbabilityCalculationMixin']:

        raise NotImplementedError

    @property
    def name(self) -> str:

        raise NotImplementedError

    set_context: Callable[[Any], 'ProbabilityCalculationMixin']
    output: Callable[[Optional[int]], CalculationValue]
