from typing import Type, List, Optional

from probability.calculations.calculation_context import CalculationContext
from probability.calculations.mixins import \
    ProbabilityCalculationMixin, OperatorMixin
from probability.calculations.calculation_types.probability_calculation import \
    ProbabilityCalculation
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class AggregatorCalculation(
    ProbabilityCalculation
):

    def __init__(
            self,
            calc_input: ProbabilityCalculationMixin,
            aggregator: Type[OperatorMixin],
            context: CalculationContext
    ):
        """
        Create a new AggregatorCalculation.

        :param calc_input: The ProbabilityCalculation to aggregate.
        :param aggregator: The OperatorMixin to use to aggregate the input.
        :param context: The CalculationContext applying to the Calculation.
        """
        self.calc_input: ProbabilityCalculationMixin = calc_input
        self.aggregator: Type[OperatorMixin] = aggregator
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List['ProbabilityCalculationMixin']:
        """
        Return the Calculation Input, as a one-element list.
        """
        return [self.calc_input]

    def output(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue:
        """
        Get the sampled output of the calculation.

        :param num_samples: Number of samples to draw.
        """
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
        """
        Return the name of the Calculation.
        """
        return self.aggregator.get_name(self.calc_input.name)
