from typing import Callable, Any, List, Optional

from probability.calculations.calculation_context import CalculationContext
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


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
        Return the existing output if it exists,
        otherwise sample it from scratch.

        :param num_samples: Number of samples to draw.
        """
        raise NotImplementedError

    def rvs(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> 'CalculationValue':
        """
        Resample the calculation output whether or not it exists.

        :param num_samples: Number of samples to draw.
        """
        self.context.clear()
        return self.output(num_samples)
