from numpy import min

from probability.calculations.operators.array_operators.array_operator import \
    ArrayOperator


class MinOperator(
    ArrayOperator,
    object
):
    """
    Operator to take the minimum of a list of distributions.
    """
    symbol = 'min'
    operator = min
    pandas_op_name = 'min'
