from operator import truediv

from probability.calculations.operators.binary_operator import BinaryOperator


class Divide(
    BinaryOperator,
    object
):
    """
    Operator to divide the values of 2 distributions.
    """
    symbol = '/'
    operator = truediv
    pandas_op_name = 'div'
