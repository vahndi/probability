from probability.calculations.calculation_types.array_calculation import \
    ArrayCalculation
from probability.calculations.operators.array_operators import MaxOperator


class Max(ArrayCalculation, object):

    operator = MaxOperator
