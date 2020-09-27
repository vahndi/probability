from typing import List

from pandas import DataFrame, Series

from probability.calculations.context import CalculationContext
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.operators import Multiply, Add
from probability.calculations.sample_calculation import SampleCalculation
from probability.calculations.value_calculation import ValueCalculation
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin


class ProbabilityCalculation(
    ProbabilityCalculationMixin,
    object
):

    context: CalculationContext

    @property
    def input_calcs(self) -> List[ProbabilityCalculationMixin]:

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
                      the context of each value will not be synced.
                      Use `sync_context` if syncing is needed.
        """
        if isinstance(other, ProbabilityCalculationMixin):
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

        from probability.calculations.binary_operator_calculation import \
            BinaryOperatorCalculation

        return BinaryOperatorCalculation(
            calc_input_1=self,
            calc_input_2=input_2,
            operator=Multiply,
            context=self.context
        )

    def __add__(self, other):

        if isinstance(other, ProbabilityCalculationMixin):
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

        from probability.calculations.binary_operator_calculation import \
            BinaryOperatorCalculation

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
            if isinstance(other, ProbabilityCalculationMixin):
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

            from probability.calculations.binary_operator_calculation import \
                BinaryOperatorCalculation

            return BinaryOperatorCalculation(
                calc_input_1=input_1,
                calc_input_2=self,
                operator=Add,
                context=self.context
            )

    def __repr__(self):

        return self.name
