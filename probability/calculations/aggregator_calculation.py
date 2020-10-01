from typing import Type, List, Optional

from probability.calculations.context import CalculationContext
from probability.calculations.mixins import \
    ProbabilityCalculationMixin, OperatorMixin
from probability.calculations.probability_calculation import \
    ProbabilityCalculation
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class AggregatorCalculation(
    ProbabilityCalculation
):

    def __init__(self,
                 calc_input: ProbabilityCalculationMixin,
                 aggregator: Type[OperatorMixin],
                 context: CalculationContext):

        self.calc_input: ProbabilityCalculationMixin = calc_input
        self.aggregator: Type[OperatorMixin] = aggregator
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List['ProbabilityCalculationMixin']:

        return [self.calc_input]

    def output(
            self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue:

        if self.context.has_object_named(self.name):
            return self.context[self.name]
        else:
            # calculate input
            if self.context.has_object_named(self.calc_input.name):
                input_ = self.context[self.calc_input.name]
            else:
                input_ = self.calc_input.output(num_samples)
                self.context[self.calc_input.name] = input_
            # calculate output
            result = self.aggregator.operate(input_)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:

        return self.aggregator.get_name(self.calc_input.name)
