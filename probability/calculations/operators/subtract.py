from operator import sub

from probability.calculations.operators.binary_operator import BinaryOperator


class Subtract(
    BinaryOperator,
    object
):
    """
    Operator to subtract the values of 2 distributions.
    """
    symbol = '-'
    operator = sub
    pandas_op_name = 'sub'
