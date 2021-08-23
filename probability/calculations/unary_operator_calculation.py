from typing import Type, List, Optional

from probability.calculations.probability_calculation import \
    ProbabilityCalculation
from probability.calculations.context import CalculationContext
from probability.calculations.mixins import OperatorMixin, \
    ProbabilityCalculationMixin
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class UnaryOperatorCalculation(
    ProbabilityCalculation
):
    """
    Calculation used to apply an operation to a Calculation e.g. the Complement.
    """
    def __init__(self,
                 calc_input: ProbabilityCalculationMixin,
                 operator: Type[OperatorMixin],
                 context: CalculationContext):
        """
        Create a new UnaryOperatorCalculation.

        :param calc_input: The input to the calculation.
        :param operator: The operator to apply to the calculation input.
        :param context: The CalculationContext.
        """
        self.calc_input: ProbabilityCalculationMixin = calc_input
        self.operator: Type[OperatorMixin] = operator
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
            result = self.operator.operate(input_)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:
        """
        Return the name of the Calculation.
        """
        return self.operator.get_name(self.calc_input.name)
