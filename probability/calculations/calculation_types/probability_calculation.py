from operator import mul, truediv, add

from pandas import DataFrame, Series

from probability.calculations.calculation_context import CalculationContext
from probability.calculations.calculation_types.sample_calculation \
    import SampleCalculation
from probability.calculations.calculation_types.simple_calculation \
    import SimpleCalculation
from probability.calculations.calculation_types.value_calculation \
    import ValueCalculation
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.calculations.operators.aggregator_operators.sum_operator \
    import SumOperator
from probability.calculations.operators import \
    AddOperator, DivideOperator, MultiplyOperator
from probability.utils import is_scalar
from probability.distributions.mixins.rv_mixins import is_rvs


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
        if is_scalar(item_2):
            input_2 = ValueCalculation(calc_input=item_2, context=context)
        elif is_rvs(item_2):
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

    from \
        probability.calculations.calculation_types.binary_operator_calculation \
        import BinaryOperatorCalculation

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
        if is_scalar(item_2):
            input_1 = ValueCalculation(calc_input=item_2, context=context)
        elif is_rvs(item_2):
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

    from \
        probability.calculations.calculation_types.binary_operator_calculation \
        import BinaryOperatorCalculation

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
            self, other, mul, MultiplyOperator
        )

    def __rmul__(self, other):

        return reverse_binary_operation(
            self, other, mul, MultiplyOperator
        )

    def __add__(self, other):
        """
        Add the Distribution to a float, distribution,
        ProbabilityCalculation, Series or DataFrame.

        :param other: The multiplier. N.B. if it is a Series or a DataFrame,
                      the context of each value will not be synced.
                      Use `sync_context` if syncing is needed.
        """
        return forward_binary_operation(
            self, other, add, AddOperator
        )

    def __radd__(self, other):

        if other == 0:
            return self
        else:
            return reverse_binary_operation(
                self, other, add, AddOperator
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
            self, other, truediv, DivideOperator
        )

    def __rtruediv__(self, other):

        return reverse_binary_operation(
            self, other, truediv, DivideOperator
        )

    def sum(self):

        from probability.calculations.calculation_types.aggregator_calculation \
            import AggregatorCalculation

        return AggregatorCalculation(
            calc_input=self,
            aggregator=SumOperator,
            context=self.context
        )

    def __repr__(self):

        return self.name
