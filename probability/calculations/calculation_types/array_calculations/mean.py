from probability.calculations.calculation_types.array_calculation import \
    ArrayCalculation
from probability.calculations.operators.array_operators import MeanOperator


class Mean(ArrayCalculation, object):

    operator = MeanOperator
