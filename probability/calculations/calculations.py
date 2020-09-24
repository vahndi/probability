from typing import Union, Type, Optional, List

from pandas import DataFrame, Series

from probability.calculations.context import CalculationContext
from probability.calculations.mixins import OperatorMixin, \
    ProbabilityCalculationMixin
from probability.calculations.operators import Multiply, Add, BinaryOperator
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin, \
    NUM_SAMPLES_COMPARISON


class SimpleCalculation(
    ProbabilityCalculationMixin,
    object
):
    pass


class ProbabilityCalculation(
    ProbabilityCalculationMixin,
    object
):

    context: CalculationContext

    @property
    def input_calcs(self) -> List['ProbabilityCalculation']:

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

    def __mul__(self, other):
        """
        Multiply the Distribution by a float, distribution,
        ProbabilityCalculation, Series or DataFrame.

        :param other: The multiplier. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced. Use
                      `sync_context` if syncing is needed.
        """
        if isinstance(other, ProbabilityCalculation):
            input_2 = other
            input_2.set_context(self.context)
        else:
            context = CalculationContext()
            if isinstance(other, float):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_2 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series) or isinstance(other, DataFrame):
                return other * self
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        return BinaryOperatorCalculation(
            calc_input_1=self,
            calc_input_2=input_2,
            operator=Multiply,
            context=self.context
        )

    def __rmul__(self, other):

        if isinstance(other, ProbabilityCalculation):
            input_1 = other
            input_1.set_context(self.context)
        else:
            context = CalculationContext()
            if isinstance(other, float):
                input_1 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_1 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series) or isinstance(other, DataFrame):
                return other * self
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        return BinaryOperatorCalculation(
            calc_input_1=input_1,
            calc_input_2=self,
            operator=Multiply,
            context=self.context
        )

    def __add__(self, other):

        if isinstance(other, ProbabilityCalculation):
            input_2 = other
            input_2.set_context(self.context)
        else:
            context = CalculationContext()
            if isinstance(other, float):
                input_2 = ValueCalculation(calc_input=other, context=context)
            elif isinstance(other, RVS1dMixin) or isinstance(other, RVSNdMixin):
                input_2 = SampleCalculation(calc_input=other, context=context)
            elif isinstance(other, Series) or isinstance(other, DataFrame):
                return other * self
            else:
                raise TypeError(
                    'other must be type Rvs1dMixin, RvsNdMixin float, '
                    'Series or DataFrame'
                )

        return BinaryOperatorCalculation(
            calc_input_1=self,
            calc_input_2=input_2,
            operator=Add,
            context=self.context
        )

    def __radd__(self, other):

        if other == 0:
            return self
        else:
            if isinstance(other, ProbabilityCalculation):
                input_1 = other
                input_1.set_context(self.context)
            else:
                context = CalculationContext()
                if isinstance(other, float):
                    input_1 = ValueCalculation(calc_input=other,
                                               context=context)
                elif isinstance(other, RVS1dMixin) or isinstance(other,
                                                                 RVSNdMixin):
                    input_1 = SampleCalculation(calc_input=other,
                                                context=context)
                elif isinstance(other, Series) or isinstance(other, DataFrame):
                    return other * self
                else:
                    raise TypeError(
                        'other must be type Rvs1dMixin, RvsNdMixin float, '
                        'Series or DataFrame'
                    )

            return BinaryOperatorCalculation(
                calc_input_1=input_1,
                calc_input_2=self,
                operator=Add,
                context=self.context
            )

    def __repr__(self):

        return self.name


class BinaryOperatorCalculation(
    ProbabilityCalculation
):

    def __init__(self,
                 calc_input_1: ProbabilityCalculation,
                 calc_input_2: ProbabilityCalculation,
                 operator: Type[BinaryOperator],
                 context: CalculationContext):

        self.calc_input_1: ProbabilityCalculation = calc_input_1
        self.calc_input_2: ProbabilityCalculation = calc_input_2
        self.operator: Type[BinaryOperator] = operator
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
            value_1_calc = not (
                isinstance(self.calc_input_1, SimpleCalculation) or
                isinstance(self.calc_input_1, int) or
                isinstance(self.calc_input_1, float)
            )
            value_2_calc = not (
                isinstance(self.calc_input_2, SimpleCalculation) or
                isinstance(self.calc_input_2, int) or
                isinstance(self.calc_input_2, float)
            )
            result = self.operator.operate(
                value_1=input_1, value_2=input_2,
                value_1_calc=value_1_calc, value_2_calc=value_2_calc
            )
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:

        if (
                not isinstance(self.calc_input_1, SimpleCalculation) and
                not isinstance(self.calc_input_1, int) and
                not isinstance(self.calc_input_1, float)
        ):
            name_1 = f'({self.calc_input_1.name})'
        else:
            name_1 = f'{self.calc_input_1.name}'

        if (
                not isinstance(self.calc_input_2, SimpleCalculation) and
                not isinstance(self.calc_input_2, int) and
                not isinstance(self.calc_input_2, float)
        ):
            name_2 = f'({self.calc_input_2.name})'
        else:
            name_2 = f'{self.calc_input_2.name}'

        return self.operator.get_name(name_1, name_2)


class UnaryOperatorCalculation(ProbabilityCalculation):

    def __init__(self,
                 calc_input: ProbabilityCalculationMixin,
                 operator: Type[OperatorMixin],
                 context: CalculationContext):

        self.calc_input: ProbabilityCalculationMixin = calc_input
        self.operator: Type[OperatorMixin] = operator
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
            result = self.operator.operate(input_)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:

        return self.operator.get_name(self.calc_input.name)


class SampleCalculation(SimpleCalculation):

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
            if isinstance(self.calc_input, RVS1dMixin):
                result = self.calc_input.rvs(num_samples)
            else:
                result = self.calc_input.rvs(num_samples, full_name=True)
            self.context[self.name] = result
            return result

    @property
    def name(self) -> str:

        return str(self.calc_input)


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
