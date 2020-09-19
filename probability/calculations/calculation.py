from typing import Union, Type, Any, Optional

from pandas import DataFrame, Series

from probability.calculations.mixins import OperatorMixin
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin, \
    NUM_SAMPLES_COMPARISON


class DistributionCalculation(object):

    executed_values: dict

    def execute(self) -> Any:

        raise NotImplementedError

    def name(self) -> str:

        raise NotImplementedError

    def __repr__(self):

        return f'DistributionCalculation: {self.name}'


CalculationValue = Union[int, float,
                         RVS1dMixin, RVSNdMixin,
                         DistributionCalculation]


class CalculationInput(object):

    valid_types = (int, float, RVS1dMixin, RVSNdMixin, Series, DataFrame)

    def __init__(self, name: str, value: CalculationValue):

        if (
                not any(isinstance(value, valid_type)
                        for valid_type in self.valid_types) and
                not isinstance(value, DistributionCalculation)
        ):
            raise TypeError(
                f'Invalid type for CalculationInput value: '
                f'{value.__class__.__name__}'
            )
        self.name: str = name
        self.value: CalculationValue = value


class BinaryOperatorCalculation(
    DistributionCalculation
):

    def __init__(self,
                 calc_input_1: CalculationInput,
                 calc_input_2: CalculationInput,
                 operator: Type[OperatorMixin]):

        self.calc_input_1: CalculationInput = calc_input_1
        self.calc_input_2: CalculationInput = calc_input_2
        self.operator: Type[OperatorMixin] = operator
        self._result = None
        self.executed_values = {}

    def execute(self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON):

        if self._result is None:

            # calculate input 1
            input_1_value = self.calc_input_1.value
            input_1_name = self.calc_input_1.name
            if type(input_1_value) in (int, float, Series, DataFrame):
                value_1 = input_1_value
            elif (
                    isinstance(input_1_value, RVS1dMixin) or
                    isinstance(input_1_value, RVSNdMixin)
            ):
                if num_samples is None:
                    num_samples = NUM_SAMPLES_COMPARISON
                value_1 = input_1_value.rvs(num_samples)
            elif isinstance(input_1_value, DistributionCalculation):
                value_1 = input_1_value.execute()
                for name, value in input_1_value.executed_values.items():
                    self.executed_values[name] = value
            else:
                raise TypeError(f'Cannot operate on type {type(input_1_value)}')
            if input_1_name not in self.executed_values:
                self.executed_values[input_1_name] = value_1

            # calculate input 2
            input_2_value = self.calc_input_2.value
            input_2_name = self.calc_input_2.name

            if type(input_2_value) in (int, float, Series, DataFrame):
                value_2 = input_2_value
            elif (
                    isinstance(input_2_value, RVS1dMixin) or
                    isinstance(input_2_value, RVSNdMixin)
            ):
                if num_samples is None:
                    num_samples = NUM_SAMPLES_COMPARISON
                value_2 = input_2_value.rvs(num_samples)
            elif isinstance(input_2_value, DistributionCalculation):
                value_2 = input_2_value.execute()
                for name, value in input_2_value.executed_values.items():
                    self.executed_values[name] = value
            else:
                raise TypeError(f'Cannot operate on type {type(input_2_value)}')
            if input_2_name not in self.executed_values:
                self.executed_values[input_2_name] = value_2

            # calculate result
            if self.name not in self.executed_values:
                self._result = self.operator.operate(
                    self.executed_values[input_1_name],
                    self.executed_values[input_2_name]
                )
                self.executed_values[self.name] = self._result
            else:
                self._result = self.executed_values[self.name]

        return self._result

    @property
    def name(self) -> str:

        if isinstance(self.calc_input_1.value, DistributionCalculation):
            name_1 = f'({self.calc_input_1.name})'
        else:
            name_1 = f'{self.calc_input_1.name}'

        if isinstance(self.calc_input_2.value, DistributionCalculation):
            name_2 = f'({self.calc_input_2.name})'
        else:
            name_2 = f'{self.calc_input_2.name}'

        return self.operator.get_name(name_1, name_2)


class UnaryOperatorCalculation(DistributionCalculation):

    def __init__(self,
                 calc_input: CalculationInput,
                 operator: Type[OperatorMixin]):

        self.calc_input: CalculationInput = calc_input
        self.operator: Type[OperatorMixin] = operator
        self._result = None
        self.executed_values = {}

    def execute(self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON):

        if self._result is None:

            # calculate input
            input_value = self.calc_input.value
            input_name = self.calc_input.name
            if type(input_value) in (int, float, Series, DataFrame):
                value = input_value
            elif (
                    isinstance(input_value, RVS1dMixin) or
                    isinstance(input_value, RVSNdMixin)
            ):
                if num_samples is None:
                    num_samples = NUM_SAMPLES_COMPARISON
                value = input_value.rvs(num_samples)
            elif isinstance(input_value, DistributionCalculation):
                value = input_value.execute()
                for name, value in input_value.executed_values.items():
                    self.executed_values[name] = value
            else:
                raise TypeError(f'Cannot operate on type {type(input_value)}')
            if input_name not in self.executed_values:
                self.executed_values[input_name] = value

            # calculate result
            if self.name not in self.executed_values:
                self._result = self.operator.operate(
                    self.executed_values[input_name],
                )
                self.executed_values[self.name] = self._result
            else:
                self._result = self.executed_values[self.name]

        return self._result

    @property
    def name(self) -> str:

        return self.operator.get_name(self.calc_input.name)
