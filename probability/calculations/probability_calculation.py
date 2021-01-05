from operator import mul, truediv, add

from pandas import DataFrame, Series

from probability.calculations.context import CalculationContext
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.operators.add import Add
from probability.calculations.operators.divide import Divide
from probability.calculations.operators.multiply import Multiply
from probability.calculations.operators.sum import Sum
from probability.calculations.sample_calculation import SampleCalculation
from probability.calculations.simple_calculation import SimpleCalculation
from probability.calculations.value_calculation import ValueCalculation
from probability.distributions.mixins.rv_mixins import RVS1dMixin, RVSNdMixin


def forward_binary_operation(
    item_1, item_2,
    builtin_operator,
    calc_operator_type
):

    if isinstance(item_2, ProbabilityCalculationMixin):
        input_2 = item_2
        input_2.set_context(item_1.context)
    else:
        context = CalculationContext()
        if isinstance(item_2, float) or isinstance(item_2, int):
            input_2 = ValueCalculation(calc_input=item_2, context=context)
        elif isinstance(item_2, RVS1dMixin) or isinstance(item_2, RVSNdMixin):
            input_2 = SampleCalculation(calc_input=item_2, context=context)
        elif isinstance(item_2, Series):
            return Series({
                key: builtin_operator(item_1, value)
                for key, value in item_2.items()
            })
        elif isinstance(item_2, DataFrame):
            return DataFrame({
                column: {key: builtin_operator(item_1, value)
                         for key, value in item_2[column].items()}
                for column in item_2.columns
            })
        else:
            raise TypeError(
                'value_2 must be type Rvs1dMixin, RvsNdMixin, int, float, '
                'Series or DataFrame'
            )

    from probability.calculations.binary_operator_calculation import \
        BinaryOperatorCalculation

    return BinaryOperatorCalculation(
        calc_input_1=item_1,
        calc_input_2=input_2,
        operator=calc_operator_type,
        context=item_1.context
    )


def reverse_binary_operation(
    item_1, item_2,
    builtin_operator,
    calc_operator_type
):
    if isinstance(item_2, ProbabilityCalculationMixin):
        input_1 = item_2
        input_1.set_context(item_1.context)
    else:
        context = CalculationContext()
        if isinstance(item_2, float) or isinstance(item_2, int):
            input_1 = ValueCalculation(calc_input=item_2, context=context)
        elif isinstance(item_2, RVS1dMixin) or isinstance(item_2, RVSNdMixin):
            input_1 = SampleCalculation(calc_input=item_2, context=context)
        elif isinstance(item_2, Series):
            return Series({
                key: builtin_operator(value, item_1)
                for key, value in item_2.items()
            })
        elif isinstance(item_2, DataFrame):
            return DataFrame({
                column: {key: builtin_operator(value, item_1)
                         for key, value in item_2[column].items()}
                for column in item_2.columns
            })
        else:
            raise TypeError(
                'item_2 must be type Rvs1dMixin, RvsNdMixin, int, float, '
                'Series or DataFrame'
            )

    from probability.calculations.binary_operator_calculation import \
        BinaryOperatorCalculation

    return BinaryOperatorCalculation(
        calc_input_1=input_1,
        calc_input_2=item_1,
        operator=calc_operator_type,
        context=item_1.context
    )


class ProbabilityCalculation(
    ProbabilityCalculationMixin,
    object
):

    context: CalculationContext

    def set_context(
            self, context: CalculationContext
    ) -> 'ProbabilityCalculation':

        self.context = context
        for input_calc in self.input_calcs:
            if not isinstance(input_calc, SimpleCalculation):
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
        return forward_binary_operation(
            self, other, mul, Multiply
        )

    def __rmul__(self, other):

        return reverse_binary_operation(
            self, other, mul, Multiply
        )

    def __add__(self, other):
        """
        Add the Distribution by a float, distribution,
        ProbabilityCalculation, Series or DataFrame.

        :param other: The multiplier. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced.
                      Use `sync_context` if syncing is needed.
        """
        return forward_binary_operation(
            self, other, add, Add
        )

    def __radd__(self, other):

        if other == 0:
            return self
        else:
            return reverse_binary_operation(
                self, other, add, Add
            )

    def __truediv__(self, other):
        """
        Divide the Distribution by a float, distribution,
        ProbabilityCalculation, Series or DataFrame.

        :param other: The multiplier. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced.
                      Use `sync_context` if syncing is needed.
        """
        return forward_binary_operation(
            self, other, truediv, Divide
        )

    def __rtruediv__(self, other):

        return reverse_binary_operation(
            self, other, truediv, Divide
        )

    def sum(self):

        from probability.calculations.aggregator_calculation import \
            AggregatorCalculation

        return AggregatorCalculation(
            calc_input=self,
            aggregator=Sum,
            context=self.context
        )

    def __repr__(self):

        return self.name
