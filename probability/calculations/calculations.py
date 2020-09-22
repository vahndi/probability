from typing import Union, Type, Optional, List

from pandas import DataFrame, Series

from probability.calculations.context import CalculationContext
from probability.calculations.mixins import OperatorMixin, \
    ProbabilityCalculationMixin
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin, \
    NUM_SAMPLES_COMPARISON


class ProbabilityCalculation(
    ProbabilityCalculationMixin,
    object
):

    context: CalculationContext

    @property
    def input_calcs(self) -> List['ProbabilityCalculation']:

        raise NotImplementedError

    def output(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue:

        raise NotImplementedError

    @property
    def name(self) -> str:

        raise NotImplementedError

    def set_context(
            self, context: CalculationContext
    ) -> 'ProbabilityCalculation':

        self.context = context
        for input_calc in self.input_calcs:
            input_calc.set_context(context)
        return self

    def __repr__(self):

        return f'DistributionCalculation: {self.name}'


class BinaryOperatorCalculation(
    ProbabilityCalculation
):

    def __init__(self,
                 calc_input_1: ProbabilityCalculation,
                 calc_input_2: ProbabilityCalculation,
                 operator: Type[OperatorMixin],
                 context: CalculationContext):

        self.calc_input_1: ProbabilityCalculation = calc_input_1
        self.calc_input_2: ProbabilityCalculation = calc_input_2
        self.operator: Type[OperatorMixin] = operator
        self.context: CalculationContext = context
        self.executed_values = {}

    @property
    def input_calcs(self) -> List[ProbabilityCalculation]:

        return [self.calc_input_1, self.calc_input_2]

    def output(
            self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue:

        if self.context.has_object_named(self.name):
            return self.context[self.name]
        else:
            # get input 1
            if self.context.has_object_named(self.calc_input_1.name):
                input_1 = self.context[self.calc_input_1.name]
            else:
                input_1 = self.calc_input_1.output(num_samples=num_samples)
                self.context[self.calc_input_1.name] = input_1
            # get input 2
            if self.context.has_object_named(self.calc_input_2.name):
                input_2 = self.context[self.calc_input_2.name]
            else:
                input_2 = self.calc_input_2.output(num_samples=num_samples)
                self.context[self.calc_input_2.name] = input_2
            # calculate output
            result = self.operator.operate(input_1, input_2)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:

        if type(self.calc_input_1) not in (ValueCalculation, SampleCalculation):
            name_1 = f'({self.calc_input_1.name})'
        else:
            name_1 = f'{self.calc_input_1.name}'

        if type(self.calc_input_2) not in (ValueCalculation, SampleCalculation):
            name_2 = f'({self.calc_input_2.name})'
        else:
            name_2 = f'{self.calc_input_2.name}'

        return self.operator.get_name(name_1, name_2)


class UnaryOperatorCalculation(ProbabilityCalculation):

    def __init__(self,
                 calc_input: ProbabilityCalculation,
                 operator: Type[OperatorMixin],
                 context: CalculationContext):

        self.calc_input: ProbabilityCalculation = calc_input
        self.operator: Type[OperatorMixin] = operator
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List['ProbabilityCalculation']:
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
            result = self.operator.operate(input_)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:

        return self.operator.get_name(self.calc_input.name)


class SampleCalculation(ProbabilityCalculation):

    def __init__(self,
                 calc_input: Union[RVS1dMixin, RVSNdMixin],
                 context: CalculationContext):

        self.calc_input: Union[RVS1dMixin, RVSNdMixin] = calc_input
        self.context: CalculationContext = context

    @property
    def input_calcs(self) -> List['ProbabilityCalculation']:

        return []

    def output(
            self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> Union[Series, DataFrame]:

        if self.context.has_object_named(self.name):
            return self.context[self.name]
        else:
            result = self.calc_input.rvs(num_samples)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:

        return str(self.calc_input)


class ValueCalculation(ProbabilityCalculation):

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
