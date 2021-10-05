from typing import Callable, Any, List, Optional

from probability.calculations.calculation_context import CalculationContext
from probability.custom_types.calculation_types import CalculationValue


class OperatorMixin(object):

    get_name: Callable[..., str]  # name of the result
    operate: Callable  # carry out the operation


class ProbabilityCalculationMixin(object):

    name: str  # name of the calculation
    context: CalculationContext
    set_context: Callable[[Any], 'ProbabilityCalculationMixin']
    # output: Callable[[Optional[int]], CalculationValue]

    @property
    def input_calcs(self) -> List['ProbabilityCalculationMixin']:
        """
        Return the inputs to the calculation.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Return the name of the calculation.
        """
        raise NotImplementedError

    def output(self, num_samples: Optional[int]) -> 'CalculationValue':
        """
        Sample the calculation output.

        :param num_samples: Number of samples to draw.
        """
        raise NotImplementedError
