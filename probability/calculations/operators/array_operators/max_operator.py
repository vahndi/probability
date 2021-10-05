from numpy import max

from probability.calculations.operators.array_operators.array_operator import \
    ArrayOperator


class MaxOperator(
    ArrayOperator,
    object
):
    """
    Operator to take the minimum of a list of distributions.
    """
    symbol = 'max'
    operator = max
    pandas_op_name = 'max'
