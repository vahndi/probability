from numpy import median

from probability.calculations.operators.array_operators.array_operator import \
    ArrayOperator


class MedianOperator(
    ArrayOperator,
    object
):
    """
    Operator to take the minimum of a list of distributions.
    """
    symbol = 'median'
    operator = median
    pandas_op_name = 'median'
