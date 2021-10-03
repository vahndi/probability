from operator import add

from probability.calculations.operators.binary_operator import BinaryOperator


class Add(
    BinaryOperator,
    object
):
    """
    Operator to add the values of 2 distributions.
    """
    symbol = '+'
    operator = add
    pandas_op_name = 'sum'
