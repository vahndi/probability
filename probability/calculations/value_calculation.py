from typing import List, Optional

from probability.calculations.simple_calculation import SimpleCalculation
from probability.calculations.context import CalculationContext
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class ValueCalculation(SimpleCalculation):

    def __init__(self,
                 calc_input: float,
                 context: CalculationContext):

        self.calc_input: float = calc_input
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List['ProbabilityCalculation']:

        return []

    def output(
            self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> float:

        if self.context.has_object_named(self.name):
            return self.context[self.name]
        else:
            self.context[self.name] = self.calc_input
            return self.calc_input

    @property
    def name(self) -> str:

        return str(self.calc_input)