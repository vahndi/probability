from probability.calculations.calculation_types.array_calculation import \
    ArrayCalculation
from probability.calculations.operators.array_operators.min_operator \
    import MinOperator


class Min(ArrayCalculation, object):

    operator = MinOperator
