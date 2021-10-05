from typing import Type, List, Optional

from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.calculation_types.probability_calculation import \
    ProbabilityCalculation
from probability.calculations.calculation_types.simple_calculation import \
    SimpleCalculation
from probability.calculations.calculation_context import CalculationContext
from probability.calculations.operators.binary_operators.binary_operator import BinaryOperator
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from probability.utils import is_scalar


class BinaryOperatorCalculation(
    ProbabilityCalculation
):
    """
    A ProbabilityCalculation that combines 2 inputs with a BinaryOperator.
    """
    def __init__(
            self,
            calc_input_1: ProbabilityCalculationMixin,
            calc_input_2: ProbabilityCalculationMixin,
            operator: Type[BinaryOperator],
            context: CalculationContext
    ):
        """
        Create a new BinaryOperatorCalculation.

        :param calc_input_1: The first Input.
        :param calc_input_2: The second Input.
        :param operator: The Binary Operator to apply.
        :param context: The CalculationContext.
        """
        self.calc_input_1: ProbabilityCalculationMixin = calc_input_1
        self.calc_input_2: ProbabilityCalculationMixin = calc_input_2
        self.operator: Type[BinaryOperator] = operator
        self.context: CalculationContext = context
        self.executed_values = {}

    @property
    def input_calcs(self) -> List[ProbabilityCalculationMixin]:
        """
        Return the Calculation Inputs as a list.
        """
        return [self.calc_input_1, self.calc_input_2]

    def output(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue:
        """
        Calculate the sampled output of the Calculation.

        :param num_samples: Number of samples to draw.
        """
        if self.context.has_object_named(self.name):
            return self.context[self.name]
        else:
            # get input 1
            if self.context.has_object_named(self.calc_input_1.name):
                input_value_1 = self.context[self.calc_input_1.name]
            else:
                input_value_1 = self.calc_input_1.output(
                    num_samples=num_samples)
                self.context[self.calc_input_1.name] = input_value_1
            # get input 2
            if self.context.has_object_named(self.calc_input_2.name):
                input_value_2 = self.context[self.calc_input_2.name]
            else:
                input_value_2 = self.calc_input_2.output(
                    num_samples=num_samples)
                self.context[self.calc_input_2.name] = input_value_2
            # calculate output
            value_1_calc = not (
                is_scalar(self.calc_input_1) or
                isinstance(self.calc_input_1, SimpleCalculation)
            )
            value_2_calc = not (
                is_scalar(self.calc_input_2) or
                isinstance(self.calc_input_2, SimpleCalculation)
            )
            result = self.operator.operate(
                value_1=input_value_1, value_2=input_value_2,
                value_1_calc=value_1_calc, value_2_calc=value_2_calc
            )
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:
        """
        Return the name of the Calculation.
        """
        if not isinstance(self.calc_input_1, SimpleCalculation):
            name_1 = f'({self.calc_input_1.name})'
        else:
            name_1 = f'{self.calc_input_1.name}'

        if not isinstance(self.calc_input_2, SimpleCalculation):
            name_2 = f'({self.calc_input_2.name})'
        else:
            name_2 = f'{self.calc_input_2.name}'

        return self.operator.get_name(name_1, name_2)
