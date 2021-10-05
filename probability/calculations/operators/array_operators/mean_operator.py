from numpy import mean

from probability.calculations.operators.array_operators.array_operator import \
    ArrayOperator


class MeanOperator(
    ArrayOperator,
    object
):
    """
    Operator to take the minimum of a list of distributions.
    """
    symbol = 'mean'
    operator = mean
    pandas_op_name = 'mean'
