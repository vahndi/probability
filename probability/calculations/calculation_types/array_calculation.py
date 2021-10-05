from typing import List, Type, Optional, Set, Union

from probability.calculations.calculation_types.value_calculation import \
    ValueCalculation
from probability.calculations.calculation_types.sample_calculation import \
    SampleCalculation
from probability.calculations.calculation_types.probability_calculation import \
    ProbabilityCalculation
from probability.calculations.calculation_context import CalculationContext
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.operators.array_operators.array_operator import \
    ArrayOperator
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON, \
    RVS1dMixin


class ArrayCalculation(ProbabilityCalculation):

    operator: Type[ArrayOperator]

    def __init__(
            self,
            *calc_inputs: Union[ProbabilityCalculationMixin, RVS1dMixin]
    ):

        contexts: Set[CalculationContext] = set()
        for calc_input in calc_inputs:
            if isinstance(calc_input, ProbabilityCalculationMixin):
                contexts.add(calc_input.context)
        if len(contexts) == 0:
            context = CalculationContext()
        elif len(contexts) == 1:
            context = list(contexts)[0]
        else:
            raise ValueError(
                'More than one context present in inputs to Minimum calculation'
            )

        array_inputs = []
        for calc_input in calc_inputs:
            if isinstance(calc_input, float):
                array_inputs.append(ValueCalculation(
                    calc_input=calc_input, context=context))
            elif isinstance(calc_input, RVS1dMixin):
                array_inputs.append(SampleCalculation(
                    calc_input=calc_input, context=context))
            else:
                raise TypeError(
                    'calc_input must be type Rvs1dMixin, float '
                    'or ProbabilityCalculation'
                )

        self.calc_inputs: List[ProbabilityCalculationMixin] = array_inputs
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List['ProbabilityCalculationMixin']:

        return self.calc_inputs

    def output(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue:
        """
        Calculate the sampled output of the Calculation.

        :param num_samples: Number of samples to draw.
        """
        input_values = []
        for calc_input in self.calc_inputs:
            if self.context.has_object_named(calc_input.name):
                input_values.append(self.context[calc_input.name])
            else:
                input_value = calc_input.output(num_samples=num_samples)
                input_values.append(input_value)
                self.context[calc_input.name] = input_value
        result = self.operator.operate(input_values)
        return result

    @property
    def name(self) -> str:
        """
        Return the name of the Calculation.
        """
        return self.operator.get_name([
            calc_input.name for calc_input in self.calc_inputs
        ])
