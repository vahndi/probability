from typing import List, Optional

from probability.calculations.calculation_types.simple_calculation import \
    SimpleCalculation
from probability.calculations.calculation_context import CalculationContext
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class ValueCalculation(SimpleCalculation):
    """
    Calculation used to wrap a float value.
    """
    def __init__(self,
                 calc_input: float,
                 context: CalculationContext):
        """
        Create a new ValueCalculation.

        :param calc_input: The float value input.
        :param context: The CalculationContext.
        """
        self.calc_input: float = calc_input
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List['ProbabilityCalculation']:
        """
        Return an empty list (Calculation has no inputs that are Calculations).
        """
        return []

    def output(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> float:
        """
        Get the sampled output of the calculation.

        :param num_samples: Number of samples to draw. Not used.
        """
        if self.context.has_object_named(self.name):
            return self.context[self.name]
        else:
            self.context[self.name] = self.calc_input
            return self.calc_input

    @property
    def name(self) -> str:
        """
        Return the name of the input value.
        """
        return str(self.calc_input)
