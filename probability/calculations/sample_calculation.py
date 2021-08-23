from typing import Union, List, Optional

from pandas import Series, DataFrame

from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.simple_calculation import SimpleCalculation
from probability.calculations.context import CalculationContext
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin, \
    NUM_SAMPLES_COMPARISON


class SampleCalculation(SimpleCalculation):
    """
    Calculation to wrap a Distribution.
    """
    def __init__(
            self,
            calc_input: Union[RVS1dMixin, RVSNdMixin],
            context: CalculationContext
    ):
        """
        Create a new Sample Calculation to wrap a Distribution.

        :param calc_input: The input Distribution.
        :param context: The context for the Calculation.
        """
        self.calc_input: Union[RVS1dMixin, RVSNdMixin] = calc_input
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List[ProbabilityCalculationMixin]:
        """
        Return an empty list (Calculation has no inputs that are Calculations).
        """
        return []

    def set_context(
            self,
            context: CalculationContext
    ) -> 'SampleCalculation':
        """
        Set the Calculation's CalculationContext.

        :param context: The CalculationContext to set.
        """
        self.context = context
        return self

    def output(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> Union[Series, DataFrame]:
        """
        Calculate the sampled output of the Calculation.

        :param num_samples: Number of samples to draw.
        """
        if self.context.has_object_named(self.name):
            return self.context[self.name]
        else:
            if isinstance(self.calc_input, RVS1dMixin):
                result = self.calc_input.rvs(num_samples)
            else:
                result = self.calc_input.rvs(num_samples, full_name=True)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:
        """
        Return the name of the Calculation's input Distribution.
        """
        return str(self.calc_input)
