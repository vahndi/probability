from typing import Union, List, Optional

from pandas import Series, DataFrame

from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.simple_calculation import SimpleCalculation
from probability.calculations.context import CalculationContext
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin, \
    NUM_SAMPLES_COMPARISON


class SampleCalculation(SimpleCalculation):

    def __init__(self,
                 calc_input: Union[RVS1dMixin, RVSNdMixin],
                 context: CalculationContext):

        self.calc_input: Union[RVS1dMixin, RVSNdMixin] = calc_input
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List[ProbabilityCalculationMixin]:

        return []

    def set_context(
            self, context: CalculationContext
    ) -> 'SampleCalculation':

        self.context = context
        return self

    def output(
            self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> Union[Series, DataFrame]:

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

        return str(self.calc_input)
