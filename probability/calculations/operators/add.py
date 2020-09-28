from operator import add

from probability.calculations.operators.binary_operator import BinaryOperator


class Add(
    BinaryOperator,
    object
):

    symbol = '+'
    operator = add
    pandas_op_name = 'sum'
